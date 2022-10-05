# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : crf.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-01-06
#   desc     :
#   refer    :
# ==========================================================================
import torch
from torchcrf import CRF as CRF_base
from entity.torch_common import to_variable, gen_mask


class CRF(CRF_base):
    def __init__(self, tag_case, device=None):
        """
        example: crf_layer = CRF(['B-abc', 'I-abc', 'O'])
        :param tag_case: which kinds of tags this CRF allows.
        :param device: torch.device('cuda:0') as default
        """
        self.tag_case = tag_case
        num_tags = len(tag_case)
        super(CRF, self).__init__(num_tags, batch_first=True)
        self.device = device or torch.device('cuda:0')

    def _validate(self, emissions: torch.Tensor, tags=None, mask=None):
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')

        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _viterbi_k_decode(self, emissions: torch.FloatTensor,
                          mask: torch.ByteTensor, k: int,
                          output_score: bool):
        # author: (ikun)wukun17s@ict.ac.cn
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_lengths, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags, k)
        score = self.start_transitions + emissions[0]
        pad_score = torch.zeros_like(score).fill_(-10000).unsqueeze(2).expand(batch_size, self.num_tags, k - 1)
        score = torch.cat([score.unsqueeze(2), pad_score], dim=-1)
        score = score.transpose(1, 2)
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, k, num_tags, 1)
            broadcast_score = score.unsqueeze(3)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1).unsqueeze(2)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags, k)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Find the maximum score over all possible current tag
            # shape: (batch_size, k, num_tags)
            next_score = next_score.contiguous().view(batch_size, -1,
                                                      self.num_tags)  # (batch_size, num_tags * k , num_tags)
            # print(next_score[0])
            next_score_k, indices = next_score.topk(k, dim=1)

            # Set score to the next score if this time step is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, k, num_tags)
            score = torch.where(mask[i].unsqueeze(1).unsqueeze(2), next_score_k, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, k, num_tags)
        score += self.end_transitions
        # Now, compute the k best path for each sample

        # shape: (batch_size, k)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []
        best_score_list = []
        for idx in range(batch_size):
            # Find the tags which are the top-k score at the last time step; this is our top-k tag
            # for the last time step
            best_score, top_k_last_tag = score[idx].flatten().topk(k, dim=0)
            best_score_list.append(best_score.cpu().detach().numpy())
            best_tags = [top_k_last_tag.cpu().numpy()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx].flatten()[best_tags[-1]]
                best_tags.append(best_last_tag.cpu().numpy())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags = [tags % self.num_tags for tags in best_tags]
            best_tags = [[int(tags[i]) for tags in best_tags] for i in range(k)]
            best_tags_list.append(best_tags)

        non_zero_tag_counts = list(map(lambda tags: sum([t != 0 for t in tags]), best_tags_list))
        best_tags_list = [bt for i, bt in enumerate(best_tags_list) if non_zero_tag_counts[i] >= 1]
        best_score_list = [bt for i, bt in enumerate(best_score_list) if non_zero_tag_counts[i] >= 1]

        if output_score:
            return best_tags_list, best_score_list
        else:
            return best_tags_list, None

    def forward_for_loss(self, emissions, mask, tags=None, reduction='token_mean'):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        loss = self.forward(emissions, tags=tags, mask=mask, reduction=reduction)
        return -loss

    def forward_for_decoding(self, emissions, mask, top_k=1, output_score=True):
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
           emissions (`~torch.Tensor`): Emission score tensor of size
               ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
               ``(batch_size, seq_length, num_tags)`` otherwise.
           mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
               if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
           List of list containing the best tag sequence for each batch.
        """
        # check if input is valid
        self._validate(emissions, mask=mask)

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        if top_k == 1:
            return self._viterbi_decode(emissions, mask=mask)
        return self._viterbi_k_decode(emissions, mask, k=top_k, output_score=output_score)

    def kwargs_for_forward(self, hidden, token_lengths, labels=None, top_k=1, reduction='token_mean'):
        """
        Usage: crf_layer(**kwargs)
        :param hidden:
        :param token_lengths:
        :param labels:
        :param reduction:
        :param is_training:
        :return:
        """
        emissions = hidden  # (batch_size, seq_length, num_tags)
        mask = to_variable(gen_mask(token_lengths), dtype='int64', device=self.device)
        mask = mask.byte()

        tags = None
        if labels is not None:
            tags = to_variable(labels, dtype='int64', device=self.device)  # (batch_size, seq_length)

        # for crf.decode() / answer for predict
        kwargs = {
            'emissions': emissions,
            'mask': mask,
            'top_k': top_k,
        }
        if tags is not None:
            # for crf() / loss for train
            kwargs.update({
                'tags': tags,
                'reduction': reduction,
            })
        return kwargs

    def forward_branch(self, emissions, mask, tags=None, top_k=1, reduction='token_mean'):
        """
        as the description for self.forward()
        :param emissions: emissions, the confidence distribution (batch_size, seq_length, num_tags)
        :param mask:
            this mask is for padding_mask
            mask = to_variable(gen_mask(inputs['token_lengths']),dtype='int64', device=self.device)
        :param tags: tags, truth tags as labels for loss-calculating
        :param reduction: none|sum|mean|token_mean
        :return:
            [train] the nll-loss in the shape of scalar (or batch_size if reduction='none')
            [infer] the tag sequence list, each element is a list of tag_indexes.
        """

        is_training = tags is not None

        if is_training:
            # scalar tensor
            ret = self.forward_for_loss(
                emissions=emissions, mask=mask, tags=tags)
        else:
            # a list of tag lists
            ret = self.forward_for_decoding(
                emissions=emissions, mask=mask, top_k=top_k)
        return ret


if __name__ == "__main__":
    pass
