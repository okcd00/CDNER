# pytorch packages
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# allennlp packages
from allennlp.nn.util import batched_index_select
from allennlp.nn import util, Activation
from allennlp.modules import FeedForward

# transformers packages
from transformers import BertTokenizer, BertPreTrainedModel, BertModel

# current repo packages
from entity.torch_common import to_variable, gen_mask
from modules.feature_fusion import FeatureFusion

import os
import json
import math
import numpy as np
from pprint import pprint

# logging 
import logging
logger = logging.getLogger('root')


class BertForEntity(BertPreTrainedModel):
    def __init__(self, config, 
                 num_ner_labels,
                 max_span_length=10,
                 head_hidden_dim=150,
                 width_embedding_dim=150,
                 args=None,):
        super().__init__(config)

        self.eps = 1e-8
        self.args = args
        self.config = config
        self.num_ner_labels = num_ner_labels
        self.max_span_length = max_span_length
        self.head_hidden_dim = head_hidden_dim
        self.width_embedding_dim = width_embedding_dim
        # self.span_filter_hp = float(args.span_filter_hp) if args else None
        # self.span_filter_strategy = str(args.span_filter_strategy) if args else None
        
        self.bert = BertModel(config)
        self.loss_fct = self.init_loss_fct(
            args.loss_function)
        self.alpha_loss_fct = None
        if args.take_alpha_loss:
            self.alpha_loss_fct = self.init_loss_fct(
                loss_fct='ce', num_labels=2) 
        self.hidden_dropout = nn.Dropout(
            config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(
            max_span_length + 1, width_embedding_dim)

        # final feature dimension
        self.inp_dim = 0  
        
        # at least we need width_embedding
        self.take_width_feature = args.take_width_feature if args else True
        if self.take_width_feature:
            self.inp_dim += width_embedding_dim

        # decide how to fusion multi-source features
        fusion_method = args.fusion_method if args else 'none'
        self.module_node = None
        self.feature_fusion = FeatureFusion(method=fusion_method)
        self.fusion_method = self.feature_fusion.method
        self.aligned_dimensions = self.feature_fusion.aligned_dimensions

        # name module
        self.take_name_module = args.take_name_module if args else True
        if self.take_name_module:
            self.init_for_name_module()

        # context module
        self.take_context_module = args.take_context_module if args else False
        if self.take_context_module:
            self.init_for_context_module()
        
        # special tokens
        self.add_pad = torch.nn.ConstantPad2d((0, 1, 0, 0), 0)  # [PAD]
        self.add_cls = torch.nn.ConstantPad2d((1, 0, 0, 0), 101)  # [CLS]
        self.add_sep = torch.nn.ConstantPad2d((0, 1, 0, 0), 102)  # [SEP]

        # classification on hiddens from the fusion layer
        self.init_for_fusion()  # also calculate self.inp_dim here
        
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=self.inp_dim,
                        num_layers=2,
                        hidden_dims=self.head_hidden_dim,  # 150
                        activations=F.relu,
                        dropout=0.2),
            nn.Linear(self.head_hidden_dim, num_ner_labels)
        )
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(p=0.1)
        self.init_weights()

    def init_loss_fct(self, loss_fct=None, num_labels=None):
        logger.info(f"Take {loss_fct} as loss function.")
        if loss_fct in ['focal', 'focal_loss']:
            from modules.focal_loss import FocalLoss
            loss_fct = FocalLoss(
                num_labels=num_labels or self.num_ner_labels,
                activation_type='sigmoid')  # focal loss
        elif loss_fct in ['label_smoothing', 'ls']:
            from modules.label_smoothing import LabelSmoothingLoss
            loss_fct = LabelSmoothingLoss(
                ignore_index=-1, reduction='sum', smoothing=0.1)
        elif loss_fct in ['nll', 'nll_loss']:
            loss_fct = torch.nn.NLLLoss(
                ignore_index=-1, reduction='sum')
        elif loss_fct in ['bce', 'bce_loss']:
            loss_fct = torch.nn.BCELoss(reduction='sum')
        else:
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss(
                ignore_index=-1, reduction='sum')
        return loss_fct

    def init_for_name_module(self):
        self.take_name_left = self.args.boundary_token in ['both', 'left', 'lef']
        self.take_name_right = self.args.boundary_token in ['both', 'right', 'rig']

        self.name_lef_vs_rig = list(map(float, [
            self.take_name_left, self.take_name_right]))
        self.name_hidden_size = self.config.hidden_size  # 768

        self.n_attention_heads = 12  # "num_attention_heads": 12
        self.n_attention_layers = 0  # we take the beginning/last k layers
        self.take_name_attn = self.n_attention_layers > 0
        self.get_attention_direction = 'begin'

    def init_for_context_module(self):
        self.take_ctx_left = self.args.boundary_token in ['both', 'left', 'lef']
        self.take_ctx_right = self.args.boundary_token in ['both', 'right', 'rig']
        self.ctx_lef_vs_rig = list(map(float, [
            self.take_ctx_left, self.take_ctx_right]))
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = True

        context_hidden_size = self.head_hidden_dim  # 150
        if self.aligned_dimensions:
            context_hidden_size = self.config.hidden_size  # 768
        self.context_hidden_size = context_hidden_size
        
        self.context_lstm = nn.LSTM(  # or nn.GRU
            input_size=self.config.hidden_size,  # 768
            hidden_size=context_hidden_size,
            num_layers=1, # dropout=0.1,
            bidirectional=True)

        if self.args.take_context_attn:
            self.query_layer = nn.Linear(context_hidden_size * 2, context_hidden_size)
            self.key_layer = nn.Linear(context_hidden_size, context_hidden_size)
            self.value_layer = nn.Linear(context_hidden_size, context_hidden_size)

    def init_for_fusion(self):
        if 'weighted' in self.feature_fusion.method:
            # set context_hidden_size = name_hidden_size
            fusion_mlp = nn.Sequential(
                FeedForward(input_dim=self.config.hidden_size,  # 768
                            num_layers=1,  
                            # num_layers=2, dropout=0.2
                            hidden_dims=self.head_hidden_dim,  # 150
                            activations=F.relu),
                nn.Linear(self.head_hidden_dim, 1)  # to scalar
            )
            if self.feature_fusion.method in ['weighted_sum', 'weighted-sum', 'gated']:
                # n_feature -> 1
                self.inp_dim += self.context_hidden_size
            else:  # 'weighted-concat'
                concat_feature_size = sum([
                    self.take_name_left,
                    self.take_name_right,
                    self.take_ctx_left,
                    self.take_ctx_right,
                    1 if self.args.take_context_attn else 0                
                ])
                self.inp_dim += self.context_hidden_size * concat_feature_size
            self.module_node = fusion_mlp
        elif 'affine' in self.feature_fusion.method:
            from modules.bi_affine import Biaffine
            bi_affine = Biaffine(n_in=self.context_hidden_size, n_out=1)
            self.module_node = bi_affine
            self.inp_dim += self.context_hidden_size  # now sum, next try {max, concat}
        else:
            self.module_node = None  # 'none' / 'concat'
            if self.take_name_module:
                self.inp_dim += self.name_hidden_size * sum([
                    self.take_name_left, self.take_name_right])
            if self.take_context_module:
                if self.args.take_context_attn:
                    self.inp_dim += self.context_hidden_size
                else:
                    self.inp_dim += self.context_hidden_size * sum([
                        self.take_ctx_left, self.take_ctx_right])
            if self.take_name_attn:
                self.inp_dim += self.n_attention_layers * self.n_attention_heads

    def token_embeddings(self, input_ids, token_type_ids=None, attention_mask=None):
        # [batch, sequence_length + 1], in case of right overflow
        am = attention_mask
        inp_ids = self.add_pad(input_ids)
        sequence_length = am.sum(-1)

        # ([batch], [batch])
        sent_indexes = torch.arange(am.shape[0], device=am.device)
        char_indexes = sequence_length.long().to(device=am.device)
        indexes = (sent_indexes, char_indexes)
        input_ids_with_sep = inp_ids.index_put_(
            indexes, torch.tensor(102, device=am.device))

        # [batch, sequence_length w/ [CLS] [SEP], embedding_size]
        embeddings = self.bert.embeddings(
            input_ids=input_ids_with_sep,  # self.add_sep(input_ids),
            token_type_ids=token_type_ids)
        return embeddings

    def context_hidden(self, embeddings, sequence_length):
        # the LSTM part
        packed = pack(embeddings, sequence_length,
                      batch_first=True, enforce_sorted=False)
        token_hidden, (h_n, c_n) = self.context_lstm(packed)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = unpack(token_hidden, batch_first=True)[0]

        # [batch, sequence_length w/ [CLS] [SEP], hidden_size * n_direction]
        token_hidden = token_hidden.view(token_hidden.shape[0], token_hidden.shape[1], 2, -1)
        # [batch, sequence_length w/ [CLS] [SEP], hidden_size] * 2 directions
        return token_hidden

    def name_attention(self, tie_hidden, sent_hidden, mask=None, do_dropout=True):
        # [batch_size, bert_emb_dim * 3]
        # => [batch_size, 1, hidden_size]
        query_layer = self.tie_query_layer(tie_hidden).unsqueeze(1)

        # [batch_size, sentence_length, name_dim]
        # => [batch_size, sentence_length, hidden_size]
        key_layer = self.tie_key_layer(sent_hidden)
        value_layer = self.tie_value_layer(sent_hidden)

        # attention_scores [batch_size, 1, sentence_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # [batch_size, 1, sentence_length]
        attention_scores = attention_scores / math.sqrt(self.name_hidden_size)

        if mask is None:  # for attention
            mask = torch.zeros_like(attention_scores, dtype=torch.uint8)

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
        # name_layer: [batch_size, 1, hidden_size] => [batch_size, hidden_size]
        # attention_probs: batch_size, 1, sentence_length
        name_layer = torch.matmul(attention_probs, value_layer).squeeze(1)
        return name_layer, attention_probs

    def context_attention(self, context_hidden, sent_hidden, mask=None, position=None, attention_mask=None,
                          share_query=False, do_dropout=True):
        """
        context_hidden: [batch_size, n_span, context_dim=hidden_size * 2]
        sent_hidden: [batch_size, n_span, sequence_length, context_dim, 2]
        position: [batch_size, num_spans], [batch_size, num_spans]
        """
        # [batch_size * n_span, context_dim=hidden_size * 2] from batched_index_select
        # => [batch_size * n_span, 1, hidden_size]
        if share_query:
            context_hidden = to_variable(
                torch.ones_like(context_hidden),
                device=self.device)
        query_layer = self.query_layer(context_hidden).unsqueeze(1)

        # for self-attention
        # sent_hidden: # [batch_size, sequence_length w/ [CLS] [SEP], n_direction=2, hidden_size] / [1, 503, 2, 150]
        # mask: [batch_size, n_span, sequence_length, n_direction=2]
        _shape = sent_hidden.shape 
        _batch_size, _sequence_length, _hidden_size = _shape[0],_shape[1], _shape[3]
        sequence_length = attention_mask.sum(-1) + 1  # [batch_size] with SEP

        if mask is None:  # for attention
            # [batch_size, n_span]
            ctx_left, ctx_right = position
            mask = torch.zeros(
                [_batch_size, ctx_left.shape[1], _sequence_length, 2], dtype=torch.uint8)
            for batch_idx in range(_batch_size):
                for i, (_l, _r) in enumerate(zip(ctx_left[batch_idx], ctx_right[batch_idx])):
                    mask[batch_idx][i][:_l+1][0] = 1
                    mask[batch_idx][i][_r: sequence_length[batch_idx]+1][1] = 1
                    # mask[i, 0, l: r] = 1  # will mask the span indexes
        mask = mask.to(device=sent_hidden.device).unsqueeze(-1)

        # sent_hidden: [batch_size, n_span=broadcast, sequence_length w/ [CLS] [SEP], n_direction=2, hidden_size]
        # mask: [batch_size, n_span, sequence_length, n_direction=2, hidden_size=broadcast]
        # => sent_hidden: [batch_size, n_span, sequence_length, context_dim]
        try:
            sent_hidden = (sent_hidden.unsqueeze(1) * mask).sum(-2)
        except:
            print(sent_hidden.unsqueeze(1).shape, mask.shape)
            raise ValueError()

        """
        for mini_batch in range(0, ctx_left.shape[1], 16):  # more time cost for less GPU capacity
            mini_mask = mask[:, mini_batch:mini_batch+16, :, :, :]
            _ctx_sent_hidden = sent_hidden.unsqueeze(1) * mini_mask
            _ctx_sent_hidden = _ctx_sent_hidden.sum(-2)
            sent_hidden.append(_ctx_sent_hidden)    
        # a list of [batch_size, mini_batch_spans, sequence_length, hidden_size] * n
        sent_hidden = torch.stack(sent_hidden, 1)
        """

        # # [batch_size, sequence_length w/ [CLS] [SEP], hidden_size, n_direction=2]
        # => [batch_size, n_span, sentence_length, hidden_size]
        key_layer = self.key_layer(sent_hidden)
        value_layer = self.value_layer(sent_hidden)

        # query [batch_size, 1, n_span, hidden_size] 
        #    => [batch_size*n_span, 1, hidden_size]
        # key   [batch_size, n_span, sentence_length, hidden_size] 
        #    => [batch_size*n_span, sentence_length, hidden_size]
        # attention_scores [batch_size, n_span, sentence_length]
        query_layer = query_layer.squeeze(1).view(-1, 1, _hidden_size)
        key_layer = key_layer.view(-1, _sequence_length, _hidden_size).transpose(-1, -2)
        # print(query_layer.shape, key_layer.shape)
        attention_scores = torch.matmul(
            query_layer, key_layer
        ).view(_batch_size, -1, _sequence_length)

        # [batch_size, n_span, sentence_length]
        attention_scores = attention_scores / math.sqrt(self.context_hidden_size)

        # mask [batch_size, n_span, sequence_length, n_direction=2]
        # attn_mask [batch_size, n_span, sentence_length]
        # attn_mask = attention_mask.unsqueeze(1).bool().cuda(self.device)
        attn_mask = (1 - mask.sum(-1).sum(-1)).bool()
        
        # attention_scores [batch_size, n_span, sentence_length]
        attention_scores = attention_scores.masked_fill(
            mask=attn_mask, value=-1e6) 

        # normalize the attention scores to probabilities.
        # attention_probs: [batch_size, n_span, sentence_length]
        attention_probs = self.softmax(attention_scores)  # <= interpret here

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        if do_dropout:
            attention_probs = self.dropout(attention_probs)

        # attention_probs: => [batch_size*n_span, 1, sentence_length] 
        # value [batch_size, n_span, sentence_length, hidden_size] 
        #    => [batch_size*n_span, sentence_length, hidden_size]
        value_layer = value_layer.view(-1, _sequence_length, _hidden_size)
        # context_logits: [batch_size, n_span, hidden_size]
        context_logits = torch.matmul(
            attention_probs.view(-1, 1, _sequence_length), 
            value_layer
        ).view(_batch_size, -1, _hidden_size)
        return context_logits, attention_probs

    def _get_span_max_pooling(self, token_hiddens, spans_start, spans_end, with_cls_sign=False):
        # [batch, n_spans, sequence_length w/ [CLS] [SEP]]
        span_mask = torch.zeros(
            [token_hiddens.shape[0], spans_start.shape[1], token_hiddens.shape[1]], 
            device=token_hiddens.device)

        # [batch, n_span]
        index_pivot = torch.arange(
            token_hiddens.shape[1], 
            device=token_hiddens.device).long()
        if with_cls_sign:
            index_pivot = index_pivot + 1
        
        # [batch, n_span, sequence_length]
        cond_lef = index_pivot >= spans_start.unsqueeze(-1)  
        cond_rig = index_pivot <= spans_end.unsqueeze(-1) 
        
        # [batch, n_spans, sequence_length, hidden_size]
        # => [batch, n_spans, hidden_size]
        span_mask[cond_lef & cond_rig] = 1
        max_pooling_hidden = (token_hiddens.unsqueeze(1) * span_mask.unsqueeze(-1)).max(-2)
        return max_pooling_hidden

    def _add_span_width_embeddings(self, spans, other_tensor=None):
        # width embeddings
        if self.take_width_feature:
            spans_width = spans[:, :, 2].view(spans.size(0), -1)
            spans_width = torch.clamp(
                spans_width, min=None, max=self.max_span_length)
            spans_width_embedding = self.width_embedding(spans_width)
            # embedding_case.append(spans_width_embedding)
            
            # print([x.shape for x in embedding_case], span_fusion_embedding.shape, self.inp_dim)
            # print(spans_width_embedding.shape, span_fusion_embedding.shape)    
            spans_embedding = torch.cat([
                spans_width_embedding,
                other_tensor, 
            ], dim=-1)
        else:
            spans_embedding = other_tensor
        return spans_embedding

    def _get_span_embeddings(self, input_ids, spans, token_type_ids=None, attention_mask=None):
        embedding_case = []
        output_attentions = self.n_attention_layers > 0
        try:
            outputs = self.bert(  # 
                input_ids=input_ids,  # [batch_size, sequence_length]
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=False
            )
            # Sequence of hidden-states at the output of the last layer of the model.
            # Last layer hidden-state of the first token of the sequence (classification token)
            # (batch_size, 2, hidden_size)
            sequence_output, pooled_output = outputs[:2]
            sequence_output = self.hidden_dropout(sequence_output)
            
            # a tuple of (batch_size, sequence_length, hidden_size)
            # (one for the output of the embeddings + one for the output of each layer)
            # hiddens = outputs[2]

            # a tuple of (batch_size, num_heads, sequence_length, sequence_length)
            # (one for each layer)
            # (batch_size, sequence_length, sequence_length, 4 * num_heads)
            if self.n_attention_layers > 0:
                if self.get_attention_direction in ['end']:
                    # last k layers
                    attentions_scores = torch.cat(outputs[2][-self.n_attention_layers:], dim=1).permute(0,2,3,1)
                else:  # ['begin']
                    # first k layers
                    attentions_scores = torch.cat(outputs[2][:self.n_attention_layers], dim=1).permute(0,2,3,1)
                # (batch_size, sequence_length * sequence_length, 4 * num_heads)
                attentions_scores = attentions_scores.view(
                    attentions_scores.size(0), -1, attentions_scores.size(-1))
                # print(attentions_scores.shape)
        
        except Exception as e:
            print(str(e))
            print(input_ids)
            print(attention_mask)
            raise Exception

        """
        spans: [batch_size, num_spans, 3]; 0: left_end, 1: right_end, 2: width
        spans_mask: (batch_size, num_spans, )
        """

        # name embedding vectors 
        token_embeddings = self.token_embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        # name hidden vectors: [batch, n_span]
        spans_start = spans[:, :, 0].view(spans.size(0), -1)
        spans_end = spans[:, :, 1].view(spans.size(0), -1)
        if output_attentions:
            # [2, 5] from 10x10 -> 25
            spans_flatten_indexes = (spans_start * input_ids.size(1) + spans_end)  
            # print(spans_flatten_indexes.shape)
            # print(spans_flatten_indexes.size())
        if self.take_name_module:
            if self.take_name_left:
                spans_start_embedding = batched_index_select(sequence_output, spans_start)
                embedding_case.append(spans_start_embedding)
            if self.take_name_right:
                spans_end_embedding = batched_index_select(sequence_output, spans_end)
                embedding_case.append(spans_end_embedding)
            if self.n_attention_layers > 0:
                # Fix: allennlp/nn/utils.py:L1161 
                # flattened_target = target.reshape(-1, target.size(-1))
                spans_attn_embedding = batched_index_select(attentions_scores, spans_flatten_indexes)
                embedding_case.append(spans_attn_embedding)

        # context hidden vectors
        if self.take_context_module:
            # [batch_size, sequence_length w/ [CLS] [SEP], n_direction=2, hidden_size]
            token_hidden = self.context_hidden(
                embeddings=token_embeddings, 
                sequence_length=attention_mask.sum(-1) + 1)  # with [SEP]
            ctx_bdy_embeddings = []
            ctx_left = spans_start - (spans_start > 0).long()  # [batch, n_span]
            ctx_right = spans_end + (spans_end > 0).long()  # [batch, n_span]
            try:
                if self.take_ctx_left:
                    context_lef = token_hidden[:, :, 0]
                    ctx_start_embedding = batched_index_select(context_lef, ctx_left)
                    # [batch_size, n_span, hidden_size]
                    ctx_bdy_embeddings.append(ctx_start_embedding)
                if self.take_ctx_right:
                    context_rig = token_hidden[:, :, 1]
                    ctx_end_embedding = batched_index_select(context_rig, ctx_right)
                    # [batch_size, n_span, hidden_size]
                    ctx_bdy_embeddings.append(ctx_end_embedding)
            except Exception as e:    
                print(str(e))
                print([x.shape for x in embedding_case])
                print(input_ids)
                print(spans_start - (spans_start > 0).long())
                print(spans_end + (spans_end > 0).long())
                raise ValueError()
            
            if self.args.take_context_attn:
                # [batch_size, n_span, hidden_size]
                ctx_bdy_attn_outputs, _ = self.context_attention(
                    context_hidden=torch.cat(ctx_bdy_embeddings, -1),  # [1, 12200, 150]
                    sent_hidden=token_hidden,  # [1, 503, 2, 150]
                    position=(ctx_left, ctx_right),  # [1, 12200]
                    attention_mask=attention_mask,  # [batch_size, sequence_length]
                    do_dropout=True)
                # [1, 12200, 150]
                embedding_case.append(ctx_bdy_attn_outputs)
            else:  # ctx_attn takes the place of ctx_embeddings
                # [batch_size * n_span, ]
                embedding_case.extend(ctx_bdy_embeddings)

        # Concatenate embeddings of left/right points and the width embedding        
        # span_fusion_embedding: a list of [batch_size, input_dim]
        # alpha: [batch_size * n_spans, 1, n_feature]
        # (biaffine) [batch_size * n_spans, 1]
        span_fusion_embedding, alpha = self.feature_fusion(
            embedding_case, 
            module_node=self.module_node, 
            return_alpha=True)

        # print([each.shape for each in embedding_case]) 
        # print(self.inp_dim)
        # print(span_fusion_embedding.shape, spans_width_embedding.shape)

        """
        dimension checklist (here contains width embeddings)
        w/ or w/o context hidden vectors.
        spans_embedding: (batch_size, num_spans, hidden_size*2+embedding_dim)
        spans_embedding: (batch_size, num_spans, hidden_size*2+head_size*2+embedding_dim)
        with weighted_sum method.
        spans_embedding: (batch_size, num_spans, embedding_dim+head_size)
        """
        return span_fusion_embedding, alpha

    def calculate_sampled_loss(
            self, logits, spans_mask, spans_ner_label, 
            loss_fct=None, sample_hp=None, sample_method=None):

        if loss_fct is None:
            loss_fct = nn.CrossEntropyLoss(reduction='sum')
        # from entity.label_smoothing import LabelSmoothingLoss
        # ls_loss = LabelSmoothingLoss(reduction='sum', smoothing=0.1))

        if spans_mask is None:
            loss = loss_fct(
                logits.view(-1, logits.shape[-1]), 
                spans_ner_label.view(-1))
            return loss

        # spans_mask is used for spans' padding
        # (batch_size * span_count)
        active_positions = spans_mask.view(-1) == 1
        # (batch_size * span_count, num_classes)
        active_logits = logits.view(
            -1, logits.shape[-1])[active_positions]
        # (batch_size * span_count)  # prob of being non-entity
        active_probs = self.softmax(active_logits)[..., 0]
        # (batch_size * span_count)
        active_labels = spans_ner_label.view(-1)[active_positions]

        # 取包含正例在内的 k 个位置计算 loss
        if None not in [sample_method, sample_hp] and sample_hp >= 0:
            positive_counts = active_labels.sum().cpu().item()
            candidates_counts = spans_mask.sum().cpu().item()

            correct_indices = active_labels > 0
            # how many negative spans will be sampled
            if sample_method in ['prop', 'rate', 'rand', 'random']:
                # k of all span candidates
                select_count = candidates_counts * sample_hp
            elif sample_method in ['count', 'counts']:
                # n samples x k count/sample
                select_count = logits.shape[0] * sample_hp
            elif sample_method in ['time', 'times', 'ratio']:
                # k times positive span counts  
                select_count = int(max(1, max(1, positive_counts) * sample_hp))
            elif sample_method in ['thres', 'threshold']:
                select_count = -1  # calculate later
            else:
                raise ValueError("Invalid sample_method", sample_method)

            # in case of candidates are not enough
            if select_count > candidates_counts:
                select_count = candidates_counts

            if select_count == 0:
                _logits = active_logits[correct_indices]
                _labels = active_labels[correct_indices]
            else:
                if sample_method in ['prop', 'rate', 'count', 'counts']:
                    # obtain spans with minimum probabilities on Non-entity
                    _, sampled_indices = torch.topk(
                        active_probs, k=select_count, 
                        dim=0, largest=False, sorted=True)
                elif sample_method in ['time', 'times', 'ratio']:
                    # obtain spans with minimum probabilities on Non-entity 
                    # but drop positive spans
                    _, sampled_indices = torch.topk(
                        active_probs + (active_labels > 0).float(), k=max(1, select_count), 
                        dim=0, largest=False, sorted=True)
                elif sample_method in ['rand', 'random']:
                    _, sampled_indices = torch.topk(
                        torch.randn_like(active_probs), k=select_count, 
                        dim=0, largest=False, sorted=True)
                elif sample_method in ['thres', 'threshold']:
                    # this method returns a mask, not indices as the ones before
                    sampled_indices = active_probs < sample_hp
                else:
                    raise ValueError("Invalid sample_method", sample_method)
                
                _logits = torch.cat([
                    active_logits[correct_indices],
                    active_logits[sampled_indices]])
                _labels = torch.cat([
                    active_labels[correct_indices],
                    active_labels[sampled_indices]])

        # print(_logits.shape)
        # print(_logits)
        # print(_labels.shape)
        # print(_labels)
        loss = loss_fct(_logits, _labels)
        return loss

    def forward(self, input_ids, spans, spans_mask, 
                spans_ner_label=None, spans_alpha_label=None,
                token_type_ids=None, attention_mask=None):
        # (batch_size, num_spans, fusion_hidden)
        spans_embedding, alpha = self._get_span_embeddings(
            input_ids, spans, token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        spans_embedding = self._add_span_width_embeddings(
            spans, spans_embedding)

        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        # (batch_size, num_spans, num_ner_labels)
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            _sample_method = 'ratio'
            _sample_hp = 20
            if hasattr(self.args, 'filtering_strategy'):
                _fs_str = self.args.filtering_strategy
                if _fs_str not in [None, 'none']:
                    _sample_method, _sample_hp = _fs_str.split('-')
                    _sample_hp = float(_sample_hp)
            loss = self.calculate_sampled_loss(
                logits=logits, 
                spans_mask=spans_mask,
                spans_ner_label=spans_ner_label, 
                loss_fct=self.loss_fct,
                sample_hp=_sample_hp, 
                sample_method=_sample_method)
            
            # alpha loss
            if (spans_alpha_label is not None) and (alpha is not None):
                active_alpha_logits = alpha.view(-1, alpha.shape[-1])
                active_alpha_labels = torch.where(
                    spans_mask.view(-1) == 1, 
                    spans_alpha_label.view(-1), 
                    torch.tensor(self.alpha_loss_fct.ignore_index).type_as(spans_alpha_label))
                # print("Alpha")
                # print(active_alpha_logits.shape)
                # print(active_alpha_logits)
                # print(active_alpha_labels.shape)
                # print(active_alpha_labels)
                # print(torch.tensor(self.alpha_loss_fct.ignore_index).type_as(spans_alpha_label))
                
                # if isinstance(self.alpha_loss_fct, LabelSmoothingLoss):
                # active_alpha_logits = torch.log(active_alpha_logits + self.eps)
                alpha_loss = self.alpha_loss_fct(
                    active_alpha_logits,  # [0. ~ 1., 1. ~ 0.]
                    active_alpha_labels)  # .float() for FocalLoss  skip_log_softmax=True for LSLoss
                # print("Loss:", loss.item(), alpha_loss.item())
                loss += alpha_loss

            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding

    def forward_origin(self, input_ids, spans, spans_mask,
                       spans_ner_label=None, spans_alpha_label=None, 
                       token_type_ids=None, attention_mask=None):
        spans_embedding, alpha = self._get_span_embeddings(
            input_ids, spans, token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        spans_embedding = self._add_span_width_embeddings(
            spans, spans_embedding)
        
        ffnn_hidden = []
        hidden = spans_embedding
        for layer in self.ner_classifier:
            hidden = layer(hidden)
            ffnn_hidden.append(hidden)
        # [batch_size, n_spans, n_classes]
        logits = ffnn_hidden[-1]

        if spans_ner_label is not None:
            if attention_mask is not None:
                active_loss = spans_mask.view(-1) == 1
                # ner loss
                active_logits = logits.view(-1, logits.shape[-1])
                active_ner_labels = torch.where(
                    active_loss,  # masking with thespans_mask
                    spans_ner_label.view(-1),  # if mask == 1
                    torch.tensor(self.loss_fct.ignore_index).type_as(spans_ner_label))
                loss = self.loss_fct(
                    active_logits, 
                    active_ner_labels)
                # print('NER')
                # print(active_logits.shape)
                # print(active_logits)
                # print(active_ner_labels.shape)
                # print(active_ner_labels)
                # alpha loss
                if False and (spans_alpha_label is not None) and (alpha is not None):
                    active_alpha_logits = alpha.view(-1, alpha.shape[-1])
                    active_alpha_labels = torch.where(
                        active_loss, spans_alpha_label.view(-1), 
                        torch.tensor(self.alpha_loss_fct.ignore_index).type_as(spans_alpha_label))
                    # print("Alpha")
                    # print(active_alpha_logits.shape)
                    # print(active_alpha_logits)
                    # print(active_alpha_labels.shape)
                    # print(active_alpha_labels)
                    # print(torch.tensor(self.alpha_loss_fct.ignore_index).type_as(spans_alpha_label))
                    
                    # if isinstance(self.alpha_loss_fct, LabelSmoothingLoss):
                    # active_alpha_logits = torch.log(active_alpha_logits + self.eps)
                    alpha_loss = self.alpha_loss_fct(
                        active_alpha_logits,  # [0. ~ 1., 1. ~ 0.]
                        active_alpha_labels)  # .float() for FocalLoss  skip_log_softmax=True for LSLoss
                    # print("Loss:", loss.item(), alpha_loss.item())
                    loss += alpha_loss
            else:  # without attention_mask (not recommended)
                loss = self.loss_fct(
                    logits.view(-1, logits.shape[-1]), 
                    spans_ner_label.view(-1))
                if spans_alpha_label is not None:
                    alpha_loss = self.alpha_loss_fct(
                        alpha.view(-1, alpha.shape[-1]), 
                        spans_alpha_label.view(-1))
                    loss += alpha_loss
            return loss, logits, spans_embedding
        else:
            return logits, spans_embedding, spans_embedding


class EntityModel():
    def __init__(self, args, num_ner_labels):
        super().__init__()

        self.args = args
        bert_model_name = args.model
        vocab_name = bert_model_name

        if args.bert_model_dir is not None:
            bert_model_name = str(args.bert_model_dir) + '/'
            # vocab_name = bert_model_name + 'vocab.txt'
            vocab_name = bert_model_name
            logger.info('Loading BERT model from {}'.format(bert_model_name))

        if args.use_albert:
            logger.info("Use Albert for testing.")
            # self.tokenizer = AlbertTokenizer.from_pretrained(vocab_name)
            from .model_albert import AlbertForEntity
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = AlbertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=args.max_span_length,
                args=args)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_name)
            self.bert_model = BertForEntity.from_pretrained(
                bert_model_name,
                num_ner_labels=num_ner_labels,
                max_span_length=args.max_span_length,
                args=args)

        self._model_device = 'cpu'
        self.move_model_to_cuda()

    def move_model_to_cuda(self):
        if not torch.cuda.is_available():
            logger.error('No CUDA found!')
            exit(-1)
        logger.info('Moving to CUDA...')
        self._model_device = 'cuda'
        self.bert_model.cuda()

        logger.info('# GPUs = %d' % (torch.cuda.device_count()))
        if torch.cuda.device_count() > 1:
            self.bert_model = torch.nn.DataParallel(self.bert_model)

    def _get_input_tensors(self, tokens, spans, spans_ner_label, spans_alpha_label=None):
        start2idx = []
        end2idx = []

        bert_tokens = []
        bert_tokens.append(self.tokenizer.cls_token)  # Add [CLS] here
        for token in tokens:
            start2idx.append(len(bert_tokens))
            sub_tokens = self.tokenizer.tokenize(token)
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens) - 1)
        bert_tokens.append(self.tokenizer.sep_token)

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(bert_tokens)
        tokens_tensor = torch.tensor([indexed_tokens])
        try:
            bert_spans = [[start2idx[span[0]], end2idx[span[1]], span[2]] for span in spans]
        except Exception as e:
            print(str(e))
            print(len(start2idx), start2idx[-20:])
            print(len(end2idx), end2idx[-20:])
            print(bert_tokens)
            bert_spans = []
            for span in spans:
                print(span)
                _tup = [start2idx[span[0]], end2idx[span[1]], span[2]]
                print(_tup)
                bert_spans.append(_tup)
        bert_spans_tensor = torch.tensor([bert_spans])
        spans_ner_label_tensor = torch.tensor([spans_ner_label])
        ret_tensors = (
            tokens_tensor, bert_spans_tensor, spans_ner_label_tensor,
        )
        if spans_alpha_label:
            spans_alpha_label_tensor = torch.tensor([spans_alpha_label])
            ret_tensors = ret_tensors + (spans_alpha_label_tensor,)
        return ret_tensors

    def _get_input_tensors_batch(self, samples_list, training=True, take_alpha_labels=False):
        tokens_tensor_list = []
        bert_spans_tensor_list = []
        spans_ner_label_tensor_list = []
        spans_alpha_label_tensor_list = []
        sentence_length = []

        max_tokens = 0
        max_spans = 0
        for sample in samples_list:
            tokens = sample['tokens']
            spans = sample['spans']
            spans_ner_label = sample['spans_label']
            
            if take_alpha_labels:
                spans_alpha_label = sample.get(
                    'spans_alpha_label', [-1 for _ in spans_ner_label])
            else:
                spans_alpha_label = None

            _input_tensors = self._get_input_tensors(
                tokens, spans, spans_ner_label, spans_alpha_label)
            if take_alpha_labels:
                tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_alpha_label_tensor = _input_tensors
                # print(list(map(lambda x: x.shape, _input_tensors)))
            else:
                tokens_tensor, bert_spans_tensor, spans_ner_label_tensor = _input_tensors

            tokens_tensor_list.append(tokens_tensor)
            bert_spans_tensor_list.append(bert_spans_tensor)
            spans_ner_label_tensor_list.append(spans_ner_label_tensor)
            if take_alpha_labels:
                spans_alpha_label_tensor_list.append(spans_alpha_label_tensor)
            assert (bert_spans_tensor.shape[1] == spans_ner_label_tensor.shape[1])
            if (tokens_tensor.shape[1] > max_tokens):
                max_tokens = tokens_tensor.shape[1]
            if (bert_spans_tensor.shape[1] > max_spans):
                max_spans = bert_spans_tensor.shape[1]
            sentence_length.append(sample['sent_length'])
        sentence_length = torch.Tensor(sentence_length)

        # apply padding and concatenate tensors
        final_tokens_tensor = None
        final_attention_mask = None
        final_bert_spans_tensor = None
        final_spans_ner_label_tensor = None
        final_spans_alpha_label_tensor = None
        final_spans_mask_tensor = None 

        if take_alpha_labels:
            tensor_zip = zip(
                tokens_tensor_list, 
                bert_spans_tensor_list, 
                spans_ner_label_tensor_list, 
                spans_alpha_label_tensor_list)
        else:
            tensor_zip = zip(
                tokens_tensor_list, 
                bert_spans_tensor_list, 
                spans_ner_label_tensor_list, 
                spans_ner_label_tensor_list  # place holder
            )

        for tokens_tensor, bert_spans_tensor, spans_ner_label_tensor, spans_alpha_label_tensor in tensor_zip:
            # padding for tokens
            num_tokens = tokens_tensor.shape[1]
            tokens_pad_length = max_tokens - num_tokens
            attention_tensor = torch.full([1, num_tokens], 1, dtype=torch.long)
            if tokens_pad_length > 0:
                pad = torch.full([1, tokens_pad_length], self.tokenizer.pad_token_id, dtype=torch.long)
                tokens_tensor = torch.cat((tokens_tensor, pad), dim=1)
                attention_pad = torch.full([1, tokens_pad_length], 0, dtype=torch.long)
                attention_tensor = torch.cat((attention_tensor, attention_pad), dim=1)

            # padding for spans
            num_spans = bert_spans_tensor.shape[1]
            spans_pad_length = max_spans - num_spans
            spans_mask_tensor = torch.full([1, num_spans], 1, dtype=torch.long)
            if spans_pad_length > 0:
                pad = torch.full([1, spans_pad_length, bert_spans_tensor.shape[2]], 0, dtype=torch.long)
                bert_spans_tensor = torch.cat((bert_spans_tensor, pad), dim=1)
                mask_pad = torch.full([1, spans_pad_length], 0, dtype=torch.long)
                spans_mask_tensor = torch.cat((spans_mask_tensor, mask_pad), dim=1)
                spans_ner_label_tensor = torch.cat((spans_ner_label_tensor, mask_pad), dim=1)
                spans_alpha_label_tensor = torch.cat((spans_alpha_label_tensor, mask_pad), dim=1)

            # update final outputs
            if final_tokens_tensor is None:
                final_tokens_tensor = tokens_tensor
                final_attention_mask = attention_tensor
                final_bert_spans_tensor = bert_spans_tensor
                final_spans_ner_label_tensor = spans_ner_label_tensor
                final_spans_alpha_label_tensor = spans_alpha_label_tensor
                final_spans_mask_tensor = spans_mask_tensor
            else:
                final_tokens_tensor = torch.cat((final_tokens_tensor, tokens_tensor), dim=0)
                final_attention_mask = torch.cat((final_attention_mask, attention_tensor), dim=0)
                final_bert_spans_tensor = torch.cat((final_bert_spans_tensor, bert_spans_tensor), dim=0)
                final_spans_ner_label_tensor = torch.cat((final_spans_ner_label_tensor, spans_ner_label_tensor), dim=0)
                final_spans_alpha_label_tensor = torch.cat((final_spans_alpha_label_tensor, spans_alpha_label_tensor), dim=0)
                final_spans_mask_tensor = torch.cat((final_spans_mask_tensor, spans_mask_tensor), dim=0)
        
        # logger.info(final_tokens_tensor)
        # logger.info(final_attention_mask)
        # logger.info(final_bert_spans_tensor)
        # logger.info(final_bert_spans_tensor.shape)
        # logger.info(final_spans_mask_tensor.shape)
        # logger.info(final_spans_ner_label_tensor.shape)

        ret_tensors = (
            final_tokens_tensor, 
            final_attention_mask, 
            final_bert_spans_tensor, 
            final_spans_mask_tensor, 
            final_spans_ner_label_tensor, 
        )
        if take_alpha_labels:
            ret_tensors = ret_tensors + (final_spans_alpha_label_tensor,)
        return ret_tensors + (sentence_length,)

    def run_batch(self, samples_list, try_cuda=True, training=True, 
                  more_span_for_detection=False, take_alpha_labels=False, confidence_level=0.5):
        # convert samples to input tensors
        _ib = self._get_input_tensors_batch(
            samples_list, training=training, take_alpha_labels=take_alpha_labels)
        if take_alpha_labels:
            tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, spans_alpha_label_tensor, _ = _ib
        else:
            tokens_tensor, attention_mask_tensor, bert_spans_tensor, spans_mask_tensor, spans_ner_label_tensor, sentence_length = _ib

        output_dict = {
            'ner_loss': 0,
        }

        if (take_alpha_labels and (spans_alpha_label_tensor is not None)):
            spans_alpha_label_tensor = spans_alpha_label_tensor.to(self._model_device)
        else:
            spans_alpha_label_tensor = None

        if training:
            self.bert_model.train()
            ner_loss, ner_logits, spans_embedding = self.bert_model(
                input_ids=tokens_tensor.to(self._model_device),
                spans=bert_spans_tensor.to(self._model_device),
                spans_mask=spans_mask_tensor.to(self._model_device),
                spans_ner_label=spans_ner_label_tensor.to(self._model_device),
                spans_alpha_label=spans_alpha_label_tensor,
                attention_mask=attention_mask_tensor.to(self._model_device),
            )
            output_dict['ner_loss'] = ner_loss.sum()
            output_dict['ner_llh'] = F.log_softmax(ner_logits, dim=-1)
        else:
            self.bert_model.eval()
            with torch.no_grad():
                ner_logits, spans_embedding, last_hidden = self.bert_model(
                    input_ids=tokens_tensor.to(self._model_device),
                    spans=bert_spans_tensor.to(self._model_device),
                    spans_mask=spans_mask_tensor.to(self._model_device),
                    spans_ner_label=None,
                    attention_mask=attention_mask_tensor.to(self._model_device),
                )
            # test for threshold adjust.
            if self.args.task.lower() in ['onto4']:
                ner_logits[..., 0] -= 3.

            # label \in {0, 1, 2, ..., n_label-1}
            if more_span_for_detection:  # more positive candidates besides 0
                _, predicted_label = ner_logits[..., 1:].max(2)  
            else:
                _, predicted_label = ner_logits.max(2)  
            
            predicted_label = predicted_label.int().cpu().numpy()
            last_hidden = last_hidden.cpu().numpy()

            predicted = []
            pred_prob = []
            hidden = []
            for i, sample in enumerate(samples_list):
                ner = []
                prob = []
                lh = []
                for j in range(len(sample['spans'])):
                    # prob.append(F.softmax(ner_logits[i][j], dim=-1).cpu().numpy())
                    pred_tag = predicted_label[i][j]
                    confidence = ner_logits[i][j].cpu().numpy()
                    prob.append(confidence)
                    if more_span_for_detection:
                        if confidence[pred_tag] > confidence_level:
                            ner.append(0)  # not-an-entity
                        else:  # the most possible tag
                            ner.append(predicted_label[i][j])  
                    else:
                        ner.append(pred_tag)
                    lh.append(last_hidden[i][j])
                predicted.append(ner)
                pred_prob.append(prob)
                hidden.append(lh)
            output_dict['pred_ner'] = predicted
            output_dict['ner_probs'] = pred_prob
            output_dict['ner_last_hidden'] = hidden

        return output_dict


if __name__ == '__main__':
    token_hiddens = torch.ones([3, 7, 5])
    spans_start = torch.ones([3, 2]).long() * 2
    spans_end = torch.ones([3, 2]).long() * 5
    
    # [batch, n_spans, sequence_length w/ [CLS] [SEP]]
    span_mask = torch.zeros(
        [token_hiddens.shape[0], spans_start.shape[1], token_hiddens.shape[1]], 
        device=token_hiddens.device)

    # [batch, n_span]
    index_pivot = torch.arange(token_hiddens.shape[1], device=token_hiddens.device).long()
    cond_lef = index_pivot >= spans_start.unsqueeze(-1)  # [batch, ]
    cond_rig = index_pivot <= spans_end.unsqueeze(-1)  # [batch, ]
    print(cond_lef.shape)
    print(cond_rig.shape)
    span_mask[cond_lef & cond_rig] = 1
    # [batch, n_spans, sequence_length, hidden_size]
    # => [batch, n_spans, hidden_size]
    max_pooling_hidden = (token_hiddens.unsqueeze(1) * span_mask.unsqueeze(-1)).max(-2)
    print(max_pooling_hidden)
