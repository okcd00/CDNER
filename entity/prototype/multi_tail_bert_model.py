# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : MultiTailBertModel.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-12-04
#   desc     : BertModel model class with shared 10 layers and 2n separate layers
# ==========================================================================
import copy
import torch
from transformers import BertModel, BertEncoder

TAIL_NAMES = ['name_aware', 'context_aware']


class MultiTailBertEncoder(BertEncoder):
    """
    A speed-up implement for different sub-tasks to share the same #0 - #9 layers,
    but different the last two layers separately.
    """
    def __init__(self, config, tail_names=None, border=10,
                 output_sharing_layers=False):
        super(MultiTailBertEncoder, self).__init__(config)
        if tail_names is None:
            tail_names = TAIL_NAMES
        self.border = border
        self.tail_names = tail_names
        self.output_sharing_layers = output_sharing_layers
        self.init_separate_layers(tail_names=tail_names)

    def init_separate_layers(self, tail_names=None, force_init=False):
        if tail_names is None:
            tail_names = self.tail_names
        for tail_name in tail_names:
            if force_init or not hasattr(self, tail_name):
                setattr(self, tail_name, copy.deepcopy(self.layer[self.border:]))

    def forward_by_tail_name(self, shared_hidden_states, attention_mask,
                             output_all_encoded_layers, tail_name):
        """

        :param shared_hidden_states: hidden states from the last shared_layer
        :param attention_mask:
        :param output_all_encoded_layers:
        :param tail_name: which layer should be applied in this tail
        :return:
        """
        extra_encoder_layers = []
        hidden_states = copy.deepcopy(shared_hidden_states)
        if tail_name == 'origin':
            target_tail_layers = self.layer[self.border:]
        else:
            target_tail_layers = getattr(self, tail_name)
        for layer_index, layer_module in enumerate(target_tail_layers):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                extra_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            extra_encoder_layers.append(hidden_states)
        return extra_encoder_layers

    def forward(self, hidden_states, attention_mask,
                output_all_encoded_layers=True, tail_names='origin'):
        """
        The output from this BertModel is a dict.
        keys are queried tail_names
        :param hidden_states:
        :param attention_mask:
        :param output_all_encoded_layers:
        :param tail_names:
        :return:
        """
        all_encoder_layers = []
        for layer_index, layer_module in enumerate(self.layer[:self.border]):
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                if self.output_sharing_layers:
                    all_encoder_layers.append(hidden_states)
        if not isinstance(tail_names, list):
            tail_names = [tail_names]

        encoder_layers = {}
        if self.output_sharing_layers:
            encoder_layers.update(
                {'sharing_layers': all_encoder_layers})
        for tail in tail_names:
            extra_layers = self.forward_by_tail_name(
                shared_hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_all_encoded_layers=output_all_encoded_layers,
                tail_name=tail)
            encoder_layers[tail] = extra_layers
        return encoder_layers


class MultiTailBertModel(BertModel):
    def __init__(self, config, tail_names=None,
                 share_pooler=True, output_sharing_layers=False):
        super(MultiTailBertModel, self).__init__(config)
        if tail_names is None:
            tail_names = TAIL_NAMES
        self.tail_names = tail_names
        self.output_sharing_layers = output_sharing_layers
        self.encoder = MultiTailBertEncoder(
            config=config, tail_names=tail_names,
            output_sharing_layers=output_sharing_layers)
        self.share_pooler = share_pooler
        if self.share_pooler:
            self.init_separate_poolers(tail_names=tail_names)
        self.apply(self.init_bert_weights)

    def intro(self):
        class_info = {
            'tail_names': self.tail_names,
            'share_pooler': self.share_pooler,
            'output_sharing_layers': self.output_sharing_layers,
        }
        return class_info

    def init_separate_poolers(self, tail_names=None, force_init=False):
        if not self.share_pooler:
            return
        if tail_names is None:
            tail_names = self.tail_names
        for tail_name in tail_names:
            pooler_name = 'pooler_' + tail_name
            if force_init or not hasattr(self, pooler_name):
                setattr(self, pooler_name, copy.deepcopy(self.pooler))

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                output_all_encoded_layers=True, tail_names='origin'):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if not isinstance(tail_names, list):
            tail_names = [tail_names]

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        try:
            embedding_output = self.embeddings(input_ids, token_type_ids)
        except Exception as e:
            print(input_ids.shape)
            print(input_ids)
            raise ValueError(str(e))

        # a dict of (tail_name, layer_list)
        encoded_layers = self.encoder(hidden_states=embedding_output,
                                      attention_mask=extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      tail_names=tail_names)
        """
        if tail_name.__len__() == 1:
            # used as the same method, return the target tail's output
            encoded_layers = encoded_layers[tail_name[0]]
            sequence_output = encoded_layers[-1]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output
        else:
        """

        encoded_layers_dict = {}
        pooled_output_dict = {}
        for tail_name in tail_names:
            _encoded_layers = encoded_layers[tail_name]
            _sequence_output = _encoded_layers[-1]
            if self.share_pooler:
                pooler_name = 'pooler_' + tail_name
                pooler = getattr(self, pooler_name)
            else:
                pooler = self.pooler
            _pooled_output = pooler(_sequence_output)
            if not output_all_encoded_layers:
                _encoded_layers = _encoded_layers[-1]
            elif self.output_sharing_layers:
                _encoded_layers = encoded_layers['sharing_layers'] + _encoded_layers
            encoded_layers_dict[tail_name] = _encoded_layers
            pooled_output_dict[tail_name] = _pooled_output
        return encoded_layers_dict, pooled_output_dict


