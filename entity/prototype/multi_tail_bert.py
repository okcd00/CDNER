# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : MultiTailBert.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-12-05
#   desc     :
# ==========================================================================
import torch
from torch import nn
import numpy as np
from torch_common import to_variable, gen_mask
from entity.prototype.multi_tail_bert_model import MultiTailBertModel, TAIL_NAMES


class MultiTailBert(nn.Module):
    def __init__(self, pretrained_path, tail_names=None,
                 device=None, bert_settings=None):
        super().__init__()
        self.device = device
        if bert_settings is None:
            bert_settings = {
                'layers': [-2, -1],  # [-4, -3, -2, -1]
                'fuse': 'cat',  # sum/cat/wsum/mean
            }
        self.tail_names = tail_names or TAIL_NAMES
        self.config = bert_settings
        self.bert = MultiTailBertModel.from_pretrained(
            pretrained_model_name=pretrained_path)

        self.hidden_size = self.bert.config.hidden_size
        self.output_sharing_layers = self.bert.output_sharing_layers
        self.output_size = 1 * self.hidden_size
        if len(bert_settings['layers']) > 1 and bert_settings['fuse'] == 'cat':
            self.output_size = len(bert_settings['layers']) * self.hidden_size

        for name, p in self.named_parameters():
            name_detected = False
            for tail_name in tail_names:
                if tail_name in name:
                    p.requires_grad = True
                    name_detected = True
                    break
            if name_detected:
                continue
            if 'bert.pooler' in name:
                p.requires_grad = True
                continue
            p.requires_grad = False

    def intro(self):
        class_info = {
            'device': self.device,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
        }
        class_info.update(self.bert.intro())
        return class_info

    def copy_params_for_separate_layers(self):
        self.bert.encoder.init_separate_layers(force_init=True)
        self.bert.init_separate_poolers(force_init=True)

    def forward(self, inputs, output_all_encoded_layers, tail_names='origin'):
        # Caution: bert use input and output data that are "batch first"
        # inputs: [sequence_length, batch_size]
        tokens_with_cls = np.vstack([101 * np.ones(inputs['tokens'].shape[1]),
                                     inputs['tokens']])
        # inputs: [batch_size, sequence_length]
        tokens_variable = to_variable(tokens_with_cls.T,
                                      dtype='int64', device=self.device)
        masks = to_variable(gen_mask(inputs['token_lengths'] + 1),  # with [CLS]
                            dtype='int64', device=self.device)
        if 'token_type_ids' in inputs:
            print('token_type used')
            token_type_ids = to_variable(inputs['token_type_ids'], device=self.device)
        else:
            token_type_ids = None

        if not isinstance(tail_names, list):
            tail_names = [tail_names]

        # [(n_layers,) batch_size, sequence_length, hidden_size]
        embeddings_dict, pool_result_dict = self.bert(
            input_ids=tokens_variable,
            token_type_ids=token_type_ids,
            attention_mask=masks,
            output_all_encoded_layers=output_all_encoded_layers,
            tail_names=tail_names)

        # [(n_layers,) batch_size, sequence_length, hidden_size]
        output_embeddings = {}
        output_cls_hiddens = {}
        for tail in tail_names:
            embeddings = embeddings_dict[tail]
            pool_result = pool_result_dict[tail]
            embeddings, cls_hiddens = self.post_process(
                embeddings, pool_result,
                output_all_encoded_layers=output_all_encoded_layers)
            output_embeddings[tail] = embeddings
            output_cls_hiddens[tail] = cls_hiddens
        # => [sequence_length - [CLS], batch_size, hidden_size], [1, batch_size, hidden_size]
        return output_embeddings, output_cls_hiddens

    def post_process(self, embeddings, pool_result, output_all_encoded_layers=False):
        # [(n_layers,) batch_size, sequence_length, hidden_size]
        # => [batch_size, sequence_length, hidden_size]
        if output_all_encoded_layers:
            # a list of embeddings for each layer.
            # self.config['layers'] = [-2, -1]
            embeddings = [embeddings[i] for i in self.config['layers']]
            if len(self.config['layers']) > 1:
                if self.config['fuse'] == 'sum':
                    embeddings = torch.stack(embeddings, dim=-1).sum(dim=-1)
                elif self.config['fuse'] == 'mean':
                    embeddings = torch.stack(embeddings, dim=-1).mean(dim=-1)
                elif self.config['fuse'] == 'cat':
                    embeddings = torch.cat(embeddings, dim=-1)
            elif len(self.config['layers']) == 1:
                embeddings = embeddings[0]
        # embeddings, pool_result = self.bert(
        # tokens_variable, attention_mask=masks, output_all_encoded_layers=True)
        # embeddings = embeddings[3]

        # => [sequence_length, batch_size, hidden_size]
        embeddings = embeddings.transpose(0, 1)
        cls_hiddens = embeddings[0]
        embeddings = embeddings[1:]
        return embeddings, cls_hiddens
