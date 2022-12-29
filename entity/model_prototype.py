# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : name_context_model.py  # [deprecated now]
#   author   : chendian / okcd00@qq.com
#   date     : 2020-11-26
#   desc     : NameContextModel Framework
# ==========================================================================
import math
import torch
from torch import nn
import numpy as np
from torch_common import to_variable, gen_mask
from modules.bi_affine import Biaffine
from modules.MultiTailBert import MultiTailBert
from modules.UTIEBert import UTIEBert, UTIECompress  # UTIECompress = Linear + ReLU
from anywhere import PRETRAINED_PATH, TARGET_CLASSES, TAIL_NAMES


class NameContextModel(nn.Module):
    def __init__(self, hidden_size=512, bert_layers=2, model_mode=None, target_types=TARGET_CLASSES,
                 tail_names=None, use_mt_bert=False, use_bi_affine=True, use_lstm_context=True,
                 share_classifiers=False, alpha_method='repr', device=None):
        super(NameContextModel, self).__init__()
        if model_mode is None:  # custom mode parameters
            model_mode = [
                3,  # mean, left bound, right bound, both bound
                2,  # joint, joint_alpha, joint_alpha_separate, alpha, alpha_separate, separate
            ]
        self.model_mode = model_mode
        self.hidden_size = hidden_size
        self.alpha_method = alpha_method
        self.use_bi_affine = use_bi_affine
        self.share_classifiers = share_classifiers
        self.device = device or torch.device('cuda:0')

        # init UTIEBert module
        self.target_types = target_types
        num_class = self.target_types.__len__()
        bert_settings = {
            'layers': list(range(-bert_layers, 0)),
            'fuse': 'cat'}

        # multi-tail bert model for speed-up
        self.use_mt_bert = use_mt_bert
        self.use_lstm_context = use_lstm_context

        if self.use_lstm_context:
            # name sub_module's bert
            self.name_bert = UTIEBert(
                pretrained_path=PRETRAINED_PATH,
                device=self.device, config=bert_settings)
            name_dim = self.name_bert.hidden_size * bert_layers  # "hidden_size": 768
            # context sub_module's bi-lstm
            n_lstm_layers = 2
            context_dim = self.hidden_size * n_lstm_layers
            # [batch, sequence_length] => [batch, sequence_length, hidden_size]
            self.context_embeddings = self.name_bert.bert.embeddings
            self.context_lstm = nn.LSTM(
                input_size=self.name_bert.hidden_size,  # "hidden_size": 768
                hidden_size=self.hidden_size,
                num_layers=n_lstm_layers,
                dropout=0.1,
                bidirectional=True)
        elif use_mt_bert:
            # share embeddings, first-n-layers and pooler
            self.bert = MultiTailBert(
                pretrained_path=PRETRAINED_PATH,
                device=self.device,
                tail_names=tail_names,
                bert_settings=bert_settings)
            self.bert.copy_params_for_separate_layers()
            name_dim = context_dim = self.bert.hidden_size * bert_layers
        else:
            # name sub_module's bert
            self.name_bert = UTIEBert(
                pretrained_path=PRETRAINED_PATH,
                device=self.device, config=bert_settings)
            name_dim = self.name_bert.hidden_size * bert_layers  # "hidden_size": 768
            # context sub_module's bert
            self.context_bert = UTIEBert(
                pretrained_path=PRETRAINED_PATH,
                device=self.device, config=bert_settings)
            context_dim = self.context_bert.hidden_size * bert_layers  # "hidden_size": 768

        # name sub_module
        self.name_trans_layer = nn.Linear(
            name_dim * 2 if self.model_mode[0] == 3 else 1,
            hidden_size)

        # context sub_module
        if self.use_lstm_context:
            query_input_dim = context_dim
        elif self.model_mode[0] == 3:
            query_input_dim = context_dim * 2
        else:
            query_input_dim = context_dim
        self.query_layer = nn.Linear(
            query_input_dim,
            hidden_size)
        self.key_layer = nn.Linear(context_dim, hidden_size)
        self.value_layer = nn.Linear(context_dim, hidden_size)

        # joint module
        if self.alpha_method.startswith('repr'):
            if self.use_bi_affine:
                self.bi_affine = Biaffine(n_in=self.hidden_size, n_out=1)
            else:  # use multi-layer compress
                self.alpha_compress = nn.ModuleList([
                    # UTIECompress(hidden_size * 2, 2),
                    UTIECompress(hidden_size * 2, hidden_size // 2),
                    nn.Linear(hidden_size // 2, 2)
                ])

        # tag classifiers
        self.name_compress = nn.ModuleList([  # for classification
            # nn.Linear(hidden_size, num_class),
            # UTIECompress(hidden_size, num_class),
            UTIECompress(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, num_class)
        ])
        if self.share_classifiers:
            self.context_compress = self.name_compress
            self.joint_compress = self.name_compress
        else:
            self.context_compress = nn.ModuleList([  # for classification
                # nn.Linear(hidden_size, num_class),
                # UTIECompress(hidden_size, num_class),
                UTIECompress(hidden_size, hidden_size // 2),
                nn.Linear(hidden_size // 2, num_class)
            ])
            self.joint_compress = nn.ModuleList([
                # nn.Linear(hidden_size, num_class),
                # UTIECompress(hidden_size, num_class),
                UTIECompress(hidden_size, hidden_size // 2),
                nn.Linear(hidden_size // 2, num_class)
            ])

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)  # the same as BertSelfAttention
        self.relu = nn.ReLU()

    def intro(self):
        class_info = {
            'use_multi_tail_bert': self.use_mt_bert,
            'use_lstm_context': self.use_lstm_context,
            'calculate_alpha_from': self.alpha_method,
            'calculate_alpha_with': 'bi-affine' if self.use_bi_affine else 'compress',
        }
        if hasattr(self, 'bert'):
            class_info.update(self.bert.intro())
        return class_info

    def bi_lstm_layer(self, embeddings, token_lengths, lstm_module=None, batch_first=True):
        """

        :param embeddings: [batch, sequence_length, hidden_size]
        :param token_lengths:
        :param lstm_module: use which nn.LSTM object
        :param batch_first: True as default
        :return: outputs, (h_n, c_n)
            outputs: [batch, sequence_length, hidden_size x 2]
            (h_n, c_n): [4, batch, hidden_size]
        """
        from torch.nn.utils.rnn import pack_padded_sequence as pack
        from torch.nn.utils.rnn import pad_packed_sequence as unpack

        if lstm_module is None:
            lstm_module = self.context_lstm

        packed = pack(embeddings, token_lengths, batch_first=batch_first, enforce_sorted=False)
        token_hidden, (h_n, c_n) = lstm_module(packed)
        # [batch, sequence_length + [CLS], hidden * n_direction]
        token_hidden = unpack(token_hidden, batch_first=batch_first)[0]
        # [batch, sequence_length, hidden * n_direction]
        return token_hidden[:, 1:, :], (h_n, c_n)

    def get_target_hidden_from_sent(self, sent_hidden, position, mode=None):
        """

        :param sent_hidden: [batch_size, sequence_length - [CLS], hidden_size=dim]
        :param position: [(l,r), (l,r), ...]
        :param mode:
        :return: target_hidden [batch_size, dim * 2]
        """
        if mode is None:
            mode = self.model_mode[0]

        # ent_embeddings: [n_entities, dim]
        if mode == 1:  # left bound
            # [batch_size, dim]
            target_hidden = torch.stack(
                [sent_hidden[i][l] for i, (l, r) in enumerate(position)])
        elif mode == 2:  # right bound
            # [batch_size, dim]
            target_hidden = torch.stack(
                [sent_hidden[i][r - 1] for i, (l, r) in enumerate(position)])
        elif mode == 3:  # left & right bound
            left_hidden = torch.stack(
                [sent_hidden[i][l] for i, (l, r) in enumerate(position)])
            right_hidden = torch.stack(
                [sent_hidden[i][r - 1] for i, (l, r) in enumerate(position)])
            # [batch_size, dim * 2]
            target_hidden = torch.cat([left_hidden, right_hidden], dim=-1)
            # [batch_size, query_length=2, dim]
            # target_hidden = torch.stack([left_hidden, right_hidden], dim=-2)
        elif mode == 4:  # separate bi-directional mode
            _shape = sent_hidden.shape
            bi_directional = sent_hidden.view(_shape[0], _shape[1], 2, -1)
            left_hidden = torch.stack(
                [bi_directional[i][l][0] for i, (l, r) in enumerate(position)])
            right_hidden = torch.stack(
                [bi_directional[i][r - 1][1] for i, (l, r) in enumerate(position)])
            # [batch_size, dim=hidden * 2]
            target_hidden = torch.cat([left_hidden, right_hidden], dim=-1)
        elif mode == 5:  # both bi-directional mode
            _shape = sent_hidden.shape
            bi_directional = sent_hidden.view(_shape[0], _shape[1], 2, -1)
            mean_hidden = bi_directional.mean(dim=-2)
            left_hidden = torch.stack(
                [mean_hidden[i][l] for i, (l, r) in enumerate(position)])
            right_hidden = torch.stack(
                [mean_hidden[i][r - 1] for i, (l, r) in enumerate(position)])
            # [batch_size, dim * 2]
            target_hidden = torch.cat([left_hidden, right_hidden], dim=-1)
            # [batch_size, query_length=2, dim]
            # target_hidden = torch.stack([left_hidden, right_hidden], dim=-2)
        else:  # if mode == 0:  # mean on span
            target_hidden = torch.stack(
                [torch.mean(sent_hidden[i][l:r], dim=-2)
                 for i, (l, r) in enumerate(position)])
        return target_hidden

    def get_hidden_vectors_from_bert(self, inputs, position, bert_module,
                                     sent_hidden=None, cls_hidden=None, mode=3):
        # bert_module: self.name_bert or self.context_bert

        # sent_hidden: [sequence_length - 1, batch_size, hidden_size]
        # cls_hidden: [1, batch_size, hidden_size]
        if sent_hidden is None or cls_hidden is None:
            sent_hidden, cls_hidden = bert_module(
                inputs,  # [sequence_length, batch_size]
                output_all_encoded_layers=True)

        # bert = torch.cat([cls_hidden, bert], dim=0)
        # => [batch_size, sequence_length - [CLS], hidden_size=dim]
        sent_hidden = sent_hidden.transpose(0, 1)

        # target_hidden: [batch_size, dim * 2]
        # cls_hidden: [batch_size, hidden_size]
        # sent_hidden: [batch_size, sequence_length, hidden_size]
        target_hidden = self.get_target_hidden_from_sent(
            sent_hidden=sent_hidden, position=position, mode=mode)

        # target_hidden, cls_hidden.squeeze(0)
        return target_hidden, sent_hidden  # logits

    def name_model_forward(self, inputs, position,
                           name_hidden=None, sent_hidden=None, mode=3):
        # [batch_size, dim * 2], [batch_size, hidden_size]
        if name_hidden is None or sent_hidden is None:
            name_bert = self.name_bert
            name_hidden, sent_hidden = self.get_hidden_vectors_from_bert(
                inputs, position, bert_module=name_bert,
                mode=mode or self.model_mode[0])

        # => [batch_size, hidden_size]
        name_repr = self.name_trans_layer(name_hidden)

        # => [batch_size, num_class]
        name_logits = self.relu(name_repr)
        for layer_module in self.name_compress:
            # more layers for joint compress
            name_logits = layer_module(name_logits)
        return name_logits, name_repr

    def context_attention(self, context_hidden, sent_hidden, mask=None, position=None,
                          share_query=False, do_dropout=True):
        # [batch_size, context_dim * 2]
        # => [batch_size, 1, hidden_size]
        if share_query:
            context_hidden = to_variable(
                torch.ones_like(context_hidden),
                device=self.device)
        query_layer = self.query_layer(context_hidden).unsqueeze(1)

        # [batch_size, sentence_length, context_dim]
        # => [batch_size, sentence_length, hidden_size]
        key_layer = self.key_layer(sent_hidden)
        value_layer = self.value_layer(sent_hidden)

        # attention_scores [batch_size, 1, sentence_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # [batch_size, 1, sentence_length]
        attention_scores = attention_scores / math.sqrt(self.hidden_size)

        # for self-attention
        # mask = torch.diag(torch.ones(sent_hidden.__len__(), dtype=torch.uint8))
        if mask is None:  # for attention
            mask = torch.zeros_like(attention_scores, dtype=torch.uint8)
        for i, (l, r) in enumerate(position):
            # mask[i][0][l + 1: r - 1] = 1
            mask[i, 0, l: r] = 1  # will mask the span indexes

        mask = mask.bool().cuda(self.device)
        attention_scores = attention_scores.masked_fill(mask=mask, value=-1e5)

        # normalize the attention scores to probabilities.
        # [batch_size, 1, sentence_length]
        attention_probs = self.softmax(attention_scores)  # <= interpret here

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if do_dropout:
            attention_probs = self.dropout(attention_probs)

        # [batch_size, 1, sentence_length] x [batch_size, sentence_length, hidden_size]
        # context_layer: [batch_size, 1, hidden_size] => [batch_size, hidden_size]
        # attention_probs: batch_size, 1, sentence_length
        context_layer = torch.matmul(attention_probs, value_layer).squeeze(1)
        return context_layer, attention_probs

    def context_model_forward(self, inputs, position,
                              context_hidden=None, sent_hidden=None, mode=3):
        # [batch_size, dim * 2], [batch_size, sequence_length - [CLS], hidden_size]
        if context_hidden is None or sent_hidden is None:
            context_bert = self.context_bert
            context_hidden, sent_hidden = self.get_hidden_vectors_from_bert(
                inputs, position, bert_module=context_bert,
                mode=mode or self.model_mode[0])

        # padding mask for attention
        input_mask = to_variable(  # [batch_size, 1, sequence_length - [CLS]]
            1 - gen_mask(inputs['token_lengths'],  # without [CLS]
                         max_len=sent_hidden.shape[1]),
            dtype='int64', device=self.device).unsqueeze(1)

        # [batch_size, hidden_size], [batch_size, sequence_length, sequence_length]
        context_repr, attn_distribution = self.context_attention(
            context_hidden, sent_hidden,
            mask=input_mask, position=position)

        # [batch_size, num_class]
        context_logits = self.relu(context_repr)
        for layer_module in self.context_compress:
            # more layers for joint compress
            context_logits = layer_module(context_logits)
        return context_logits, context_repr, attn_distribution

    def context_model_forward_bi_lstm(self, inputs, position, mode=4):
        # inputs['tokens']: [sequence_length, batch_size]
        # => [sequence_length + 1, batch_size]
        tokens_with_cls = np.vstack(
            [101 * np.ones(inputs['tokens'].shape[1]), inputs['tokens']])
        # => [batch_size, sequence_length + 1]
        tokens_variable = to_variable(
            tokens_with_cls.T, dtype='int64', device=self.device)
        token_lengths_variable = to_variable(
            inputs['token_lengths'] + 1, dtype='int64', device=self.device)

        # [batch, sequence_length + 1, embedding_size=768]
        sent_embeddings = self.context_embeddings(tokens_variable)
        # [batch, sequence_length, hidden x 2]
        sent_hidden, (h_n, c_n) = self.bi_lstm_layer(
            embeddings=sent_embeddings,
            token_lengths=token_lengths_variable,
            lstm_module=self.context_lstm,
            batch_first=True)
        context_hidden = self.get_target_hidden_from_sent(
            sent_hidden=sent_hidden,
            position=position,
            mode=mode)

        # padding mask for attention
        input_mask = to_variable(  # [batch_size, 1, sequence_length - [CLS]]
            1 - gen_mask(inputs['token_lengths'],  # without [CLS]
                         max_len=sent_hidden.shape[1]),
            dtype='int64', device=self.device).unsqueeze(1)

        # [batch_size, hidden_size], [batch_size, sequence_length, sequence_length]
        context_repr, attn_distribution = self.context_attention(
            context_hidden, sent_hidden,
            mask=input_mask, position=position)

        # [batch_size, num_class]
        context_logits = self.relu(context_repr)
        for layer_module in self.context_compress:
            # more layers for joint compress
            context_logits = layer_module(context_logits)
        return context_logits, context_repr, attn_distribution

    @staticmethod
    def joint_model_forward_on_logits(name_logits, context_logits, alpha):
        # the last dimension for alpha can be 1 or 2
        logits = context_logits * alpha[:, :1] + name_logits * alpha[:, -1:]
        # [batch_size, num_class]
        return logits

    def joint_model_forward_on_repr(self, name_repr, context_repr, alpha):
        # the last dimension for alpha can be 1 or 2
        # [batch_size, hidden_size]
        joint_repr = context_repr * alpha[:, :1] + name_repr * alpha[:, -1:]
        for layer_module in self.joint_compress:
            # more layers for joint compress
            joint_repr = layer_module(joint_repr)
        # [batch_size, num_class]
        return joint_repr

    def joint_model_forward(self, name_vector, context_vector, alpha, method='logits'):
        if method.startswith('logits'):
            # [batch_size, num_class] => [batch_size, num_class]
            return self.joint_model_forward_on_logits(
                name_logits=name_vector,
                context_logits=context_vector,
                alpha=alpha)
        elif method.startswith('repr'):
            # [batch_size, hidden_size] => [batch_size, num_class]
            return self.joint_model_forward_on_repr(
                name_repr=name_vector,
                context_repr=context_vector,
                alpha=alpha)
        else:
            raise ValueError("Invalid method for joint_model_forward(): ", method)

    def calculate_alpha_with_maximum(self, name_vector, context_vector):
        # [batch_size, num_class] => [batch_size]
        max_context_logit, max_context_index = torch.max(context_vector, dim=-1)
        max_name_logit, max_name_index = torch.max(name_vector, dim=-1)
        # [batch_size, 2]
        alpha = self.softmax(
            torch.stack([max_context_logit, max_name_logit], dim=-1))
        return alpha

    def calculate_alpha_with_compress(self, name_vector, context_vector):
        # [batch_size, hidden_size * 2]
        repr_case = torch.cat([context_vector, name_vector], dim=-1)
        # [batch_size, hidden_size * 2] => [batch_size, 2]
        for layer_module in self.alpha_compress:
            repr_case = layer_module(repr_case)
        alpha = self.softmax(repr_case)
        return alpha

    def calculate_alpha_with_biaffine(self, name_vector, context_vector):
        # [batch_size, seq_len=1, n_in=hidden_size]
        context_vector = context_vector.unsqueeze(1)
        name_vector = name_vector.unsqueeze(1)
        # [batch_size, (n_out=1,) seq_len=1, seq_len=1]
        alpha = self.bi_affine(context_vector, name_vector)
        # [batch_size, 2]
        alpha = self.sigmoid(alpha.view([alpha.shape[0], 1]))
        return torch.cat([1. - alpha, alpha], dim=-1)

    def forward(self, inputs, position, output_context_attn=False):
        """

        :param inputs: a dict with tokens, token_lengths as keys
        :param position:
        :param output_context_attn:
        :return:
        """

        if self.use_mt_bert:
            content_hidden, cls_hidden = self.bert(
                inputs,  # [sequence_length, batch_size]
                output_all_encoded_layers=True,
                tail_names=TAIL_NAMES)
            name_hidden, sent_hidden = self.get_hidden_vectors_from_bert(
                inputs, position, bert_module=None,
                sent_hidden=content_hidden['name_aware'],
                cls_hidden=cls_hidden['name_aware'],
                mode=self.model_mode[0])
            # [batch, num_class], [batch_size, hidden_size]
            name_logits, name_repr = self.name_model_forward(
                inputs, position,
                name_hidden=name_hidden,
                sent_hidden=sent_hidden)
            context_hidden, sent_hidden = self.get_hidden_vectors_from_bert(
                inputs, position, bert_module=None,
                sent_hidden=content_hidden['context_aware'],
                cls_hidden=cls_hidden['context_aware'],
                mode=self.model_mode[0])
            # [batch, num_class], [batch_size, hidden_size]
            context_logits, context_repr, attention_dist = self.context_model_forward(
                inputs, position,
                context_hidden=context_hidden,
                sent_hidden=sent_hidden)
        elif self.use_lstm_context:
            # [batch, num_class], [batch_size, hidden_size]
            name_logits, name_repr = self.name_model_forward(
                inputs=inputs, position=position)
            # [batch, num_class], [batch_size, hidden_size]
            context_logits, context_repr, attention_dist = self.context_model_forward_bi_lstm(
                inputs=inputs, position=position)
        else:
            # [batch, num_class], [batch_size, hidden_size]
            name_logits, name_repr = self.name_model_forward(
                inputs=inputs, position=position)
            # mask mention tokens, proved to be un-useful
            # inputs = add_mask_for_context_inputs(inputs, position)
            context_logits, context_repr, attention_dist = self.context_model_forward(
                inputs=inputs, position=position)

        if self.alpha_method.startswith('repr'):
            name_vector = name_repr
            context_vector = context_repr
            if self.use_bi_affine:
                alpha_fn = self.calculate_alpha_with_biaffine
            else:
                alpha_fn = self.calculate_alpha_with_compress
        else:
            name_vector = name_logits
            context_vector = context_logits
            alpha_fn = self.calculate_alpha_with_maximum

        # [batch_size, 2] from 0 to 1
        alpha_prediction = alpha_fn(name_vector, context_vector)
        alpha_prediction = to_variable(
            alpha_prediction, device=self.device)

        # [batch, num_class]
        logits = self.joint_model_forward(
            name_vector, context_vector,
            alpha=alpha_prediction,
            method=self.alpha_method)
        # self.joint_model_forward_on_repr(
        #   name_repr, context_repr, alpha.unsqueeze(-1))

        if output_context_attn:
            # [batch_size, key_length=1, sentence_length]
            context_attention = attention_dist
            return logits, name_logits, context_logits, alpha_prediction, context_attention
        return logits, name_logits, context_logits, alpha_prediction


if __name__ == "__main__":
    pass
