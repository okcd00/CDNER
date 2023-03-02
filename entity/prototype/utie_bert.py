# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : UTIEBert.py
#   origin   : cyx
#   author   : chendian / okcd00@qq.com
#   date     : 2020-11-02
#   desc     : UTIEBert with minor modifies.
# ==========================================================================
import torch
from torch import nn
import numpy as np
from torch_common import to_variable, gen_mask
from transformers import BertModel


class UTIECompress(nn.Module):
    # origin UTIECompress from utie.model
    def __init__(self, input_size, output_size, force=False):
        """
        进行一个线性变换把 input_size 变到 output_size
        :param input_size:
        :param output_size:
        :param force: =False, 如果 input=output，就不进行变换。=True 任何情况都进行变换
        """
        super(UTIECompress, self).__init__()
        self.trans = force or (input_size != output_size)
        if self.trans:
            self.linear = nn.Linear(input_size, output_size)
            self.activation = nn.ReLU()

    def forward(self, tensor):
        if self.trans:
            return self.activation(self.linear(tensor))
        else:
            return tensor


class UTIEBert(nn.Module):
    def __init__(self, pretrained_path, device=None, config=None):
        super(UTIEBert, self).__init__()
        self.device = device
        if config is None:
            config = {
                'layers': [-2, -1],  # [-4, -3, -2, -1]
                'fuse': 'cat',  # sum/cat/wsum/mean
            }
        print("bert-module config:", config)
        self.config = config
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.hidden_size = self.bert.config.hidden_size
        self.output_size = 1 * self.hidden_size
        if len(config['layers']) > 1 and config['fuse'] == 'cat':
            self.output_size = len(config['layers']) * self.hidden_size

        for name, p in self.named_parameters():
            if 'bert.encoder.layer.10' in name or 'bert.encoder.layer.11' in name or \
                    'bert.pooler' in name:
                p.requires_grad = True
                continue
            p.requires_grad = False

    def forward(self, inputs, output_all_encoded_layers, drop_cls_token=True):
        # Caution: bert use input and output data that are "batch first"
        # inputs: [sequence_length, batch_size]
        tokens_with_cls = np.vstack([101 * np.ones(inputs['tokens'].shape[1]), inputs['tokens']])
        # inputs: [batch_size, sequence_length]
        tokens_variable = to_variable(tokens_with_cls.T, dtype='int64', device=self.device)
        masks = to_variable(gen_mask(inputs['token_lengths'] + 1), dtype='int64', device=self.device)
        if 'token_type_ids' in inputs:
            print('token_type used')
            token_type_ids = to_variable(inputs['token_type_ids'], device=self.device)
            embeddings, pool_result = self.bert(tokens_variable, token_type_ids=token_type_ids, attention_mask=masks,
                                                output_all_encoded_layers=output_all_encoded_layers)
        else:
            # [(n_layers,) batch_size, sequence_length, hidden_size]
            embeddings, pool_result = self.bert(tokens_variable, attention_mask=masks,
                                                output_all_encoded_layers=output_all_encoded_layers)
        # [(n_layers,) batch_size, sequence_length, hidden_size]
        embeddings, cls_hiddens = self.post_process(
            embeddings, pool_result,
            drop_cls_token=drop_cls_token,
            output_all_encoded_layers=output_all_encoded_layers)
        # => [sequence_length - [CLS], batch_size, hidden_size], [1, batch_size, hidden_size]
        return embeddings, cls_hiddens

    def post_process(self, embeddings, pool_result, output_all_encoded_layers=False, drop_cls_token=True):
        # [(n_layers,) batch_size, sequence_length, hidden_size]
        # => [batch_size, sequence_length, hidden_size]
        if output_all_encoded_layers:
            # a list of embeddings for each layer.
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
        if drop_cls_token:
            embeddings = embeddings[1:]
        return embeddings, cls_hiddens

