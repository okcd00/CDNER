# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : span_filter.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-07-02
#   desc     : runners are simple LSTM-CRF / BERT-CRF models.
# ==========================================================================
import os
import copy
import json
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


def load_runners():
    # span filter methods from doc_ner
    import sys
    sys.path.append('/home/chendian/UTIE/')
    sys.path.append('/home/chendian/doc_ner/')
    # runners are simple LSTM-CRF / BERT-CRF models. 
    # (for online-prediction, not provided here)
    from modules.runner_binary import Runner as bin_runner
    from modules.runner_boundary import Runner as bdy_runner
    # from modules.runner_crf import Runner as crf_runner


def flatten(nested_list, unique=False):
    ret = [elem for sub_list in nested_list for elem in sub_list]
    if unique:
        return list(set(ret))
    return ret


class SpanFilter(object):
    PUNC_LIST = "，；。？！……"

    def __init__(self, ner_label2id=None, max_span_length=25, 
                 drop_with_punc=True, real_time_predict=False, 
                 evaluate_with_rate=False, filter_threshold=0.1, 
                 filter_method=np.max, method='threshold', 
                 boundary_only_mode='none'):
        super().__init__()
        self.debug = False
        self.method = self.method_mapping(method)
        self.ner_label2id = ner_label2id
        self.max_span_length = max_span_length
        self.drop_with_punc = drop_with_punc
        self.real_time_predict = real_time_predict

        self.filter_method = filter_method
        self.filter_threshold = filter_threshold
        self.evaluate_with_rate = evaluate_with_rate
        self.boundary_only_mode = boundary_only_mode
        
        self.include_positive = False
        self.random_select_spans = False
        self.sg_model = None

    def method_mapping(self, method):
        method = method.lower()
        method = {
            'none': 'threshold',
            'thres': 'threshold',
            'time': 'times',
            'count': 'count',
        }.get(method, method)
        # threshold / rate / times / count
        return method

    def load_pure_api(self, args, load_model_dir):
        if self.real_time_predict:
            from pure_api import PureApi
            self.sg_model = PureApi(
                args=args, load_model_dir=load_model_dir)
            return self.sg_model

    def load_runner_binary(self, model_path, results=None, do_predict=False):
        # deprecated
        from modules.runner_binary import Runner as bin_runner
        r = bin_runner(hidden_method='bert', model_path=model_path)
        print(r.model_path)

        if do_predict:
            print("Now predicting on {}".format(r.test_path))
            results = r.predict_on_test_loader()
            # ret_samples from r.predict has the keys 'tag_scores'

    def prf_on_token_level_results(self, results, thres=1e-3):
        output_log = []
        # from span_filtering.scripts import calculate_prf_on_token_level_results
        # print(calculate_prf_on_token_level_results(results))

        # from span_filtering.scripts import calculate_tur_from_samples_boundary
        # h, t, c = calculate_tur_from_samples_boundary(
        #     results, thres, 'boundary-both')
        # print('(θ={}) {} / {} 即 {:.1f}% / {} 倍'.format(
        #       thres, h, c, h / t * 100, c // h))
        return output_log

    def distill_span_candidates(self, selected_span_score, positive_span_count):
        # calculate how many spans will be left
        selected_indexes = np.argsort(selected_span_score)  # [2,3,1,1] -> [2,3,0,1]
        random_select = self.random_select_spans

        if self.method in ['time', 'times']:
            self.evaluate_with_rate = True
        if (not self.include_positive) and self.evaluate_with_rate:  
            # from bisect import bisect_left
            # positive_indexes is 0 when self.include_positive is False
            select_counts = int(len(selected_span_score) * 0.1)  # fix a hyperparam 0.1 or 0.2 here (rate mode)
        elif 'rate' in self.method:
            select_counts = int(len(selected_indexes) * float(self.filter_threshold))
        elif 'time' in self.method:
            # How many times the number of span selected is the number of positive ones
            # we don't know the count of positive spans in dev or test datasets
            select_counts = int(positive_span_count * self.filter_threshold)
        elif 'thres' in self.method:
            # 'threshold' / 'none'
            select_counts = sum(map(lambda _s: _s >= self.filter_threshold, selected_span_score))
        elif 'count' in self.method:  # negative spans' count 
            select_counts = int(self.filter_threshold) + positive_span_count
        else: 
            raise ValueError(f"invalid filter method: {self.method}")
        if self.debug:
            print(f"{self.method} ({self.filter_threshold}): "
                    f"Select top-{select_counts} spans from {len(selected_indexes)} selected_indexes.")
        
        # take top spans from firstly-selected span candidates
        # select_counts must be greater than 0
        select_counts = min(max(1, select_counts), len(selected_indexes))  
        if self.debug:
            print(f"select {select_counts} from {len(selected_indexes)}")
        if random_select and self.include_positive:  # sample method, in training method.
            selected_indexes = random.sample(selected_indexes.tolist(), select_counts)
        else:  # drop low-confidence only
            if select_counts == len(selected_indexes):
                selected_indexes = selected_indexes.tolist()
            else:
                selected_indexes = selected_indexes[-select_counts:].tolist()
        return selected_indexes

    def target_enumerating(self, sample, sent):
        # a list of [batch_size, input_dim]
        sent_ner = {}
        positive_indexes = []
        for ner in sent.ner:
            positive_indexes.append(ner.span.span_sent)
            sent_ner[ner.span.span_sent] = ner.label
        
        span2id = {}
        num_cover = 0
        offset = sample['sent_start']

        sample['spans'] = []
        sample['spans_label'] = []
        
        # extend spans in ner
        for (i, j) in sent_ner:
            # longer width share the longest width embedding
            span_tup = (i + offset, j + offset,  
                        min(j - i + 1, self.max_span_length))
            sample['spans'].append(span_tup)
            span2id[(i, j)] = len(sample['spans']) - 1
            sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
            num_cover += 1

        # in case of empty ones.
        if len(sample['spans']) == 0:  
            span_tup = (offset, offset, 1)
            sample['spans'].append(span_tup)
            sample['spans_label'].append(0)

        sample['positive_labels_count'] = len(positive_indexes)
        sample['positive_labels_cover'] = num_cover
        assert len(sample['spans']) == len(sample['spans_label'])
        return sample

    def default_enumerating(self, sample, sent):
        # a list of [batch_size, input_dim]
        sent_ner = {}
        positive_indexes = []
        for ner in sent.ner:
            positive_indexes.append(ner.span.span_sent)
            if self.boundary_only_mode in ['span']:
                sent_ner[ner.span.span_sent] = '实体'
            else:
                sent_ner[ner.span.span_sent] = ner.label
        
        span2id = {}
        num_cover = 0
        offset = sample['sent_start']

        sample['spans'] = []
        sample['spans_label'] = []
        for i in range(len(sent.text)):
            for j in range(i, min(len(sent.text), i + self.max_span_length)):
                if self.drop_with_punc and sent.text[j] in self.PUNC_LIST:
                    break  # entity spans do not cover punctuations
                span_tup = (i + offset, j + offset, j - i + 1)
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                if (i, j) not in sent_ner:
                    sample['spans_label'].append(0)
                else:
                    sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                    if self.include_positive:
                        del sent_ner[(i, j)]
                    num_cover += 1
        
        # extend spans not mentioned (only in training mode)
        if self.include_positive:
            for (i, j) in sent_ner:
                # longer width share the longest width embedding
                span_tup = (i + offset, j + offset,  
                            min(j - i + 1, self.max_span_length))
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                num_cover += 1

        # in case of empty ones.
        if len(sample['spans']) == 0:  
            span_tup = (offset, offset, 1)
            sample['spans'].append(span_tup)
            sample['spans_label'].append(0)

        sample['positive_labels_count'] = len(positive_indexes)
        sample['positive_labels_cover'] = num_cover
        assert len(sample['spans']) == len(sample['spans_label'])
        return sample

    def enumerating_one_char_spans(self, sample, sent):
        # a list of [batch_size, input_dim]
        
        sent_ner = {}
        for ner in sent.ner:
            lef, rig = ner.span.span_sent
            if self.boundary_only_mode in ['bdy']:
                sent_ner[(lef, lef)] = 'B-实体'
                if lef != rig:
                    sent_ner[(rig, rig)] = 'E-实体'
                else:
                    sent_ner[(rig, rig)] = 'S-实体'  # overwrite
            elif self.boundary_only_mode in ['bin']:
                for idx in range(lef, rig+1):
                    sent_ner[(idx, idx)] = 'IN-实体'  # in, not inner
            else:  
                sent_ner[(lef, lef)] = f'B-{ner.label}'
                if lef != rig:
                    sent_ner[(rig, rig)] = f'E-{ner.label}'
                else:
                    sent_ner[(rig, rig)] = f'S-{ner.label}'  # overwrite

        span2id = {}
        num_cover = 0
        offset = sample['sent_start']

        sample['spans'] = []
        sample['spans_label'] = []
        for i in range(len(sent.text)):
            j = i
            # if self.drop_with_punc and sent.text[j] in self.PUNC_LIST:
            #     break  # entity spans do not cover punctuations
            span_tup = (i + offset, j + offset, j - i + 1)
            sample['spans'].append(span_tup)
            span2id[(i, j)] = len(sample['spans']) - 1
            if (i, j) not in sent_ner:
                sample['spans_label'].append(0)
            else:
                sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                num_cover += 1

        if len(sample['spans']) == 0:  # in case of empty ones.
            span_tup = (sample['sent_start'], sample['sent_start'], 1)
            sample['spans'].append(span_tup)
            sample['spans_label'].append(0)

        sample['positive_labels_count'] = len(sent_ner)
        sample['positive_labels_cover'] = num_cover
        return sample

    def enumerating_with_tag_score(self, sample, sent):
        # a list of [batch_size, input_dim]
        sent_ner = {}
        tag_score = {}
        positive_indexes = []

        text_len = len(sent.text)
        for ner in sent.ner:
            lef, rig = ner.span.span_sent
            sent_ner[(lef, rig)] = ner.label
            positive_indexes.append((lef, rig))
            if self.include_positive:
                tag_score[(lef, rig)] = 1.

        num_cover = 0
        sent_scores = sent.scores if sent.scores else []  # if not available
        offset = sample['sent_start']  # for context
        # doc_offset = sample['sent_start_in_doc']  # for document
        for i, score in enumerate(sent_scores):
            if 'bin' in self.boundary_only_mode:
                tag_score[i] = score
            elif 'bdy' in self.boundary_only_mode:
                lef_score, rig_score = score
                tag_score[i] = (lef_score, rig_score)
        
        span2id = {}
        sample['spans'] = []
        sample['spans_label'] = []
        selected_span_score = []
        
        # enumerate span candidates
        for i in range(text_len):
            for j in range(i, min(len(sent.text), i + self.max_span_length)):
                if self.drop_with_punc and sent.text[j] in self.PUNC_LIST:
                    break  # entity spans do not cover punctuations
                # if 'bin' in self.boundary_only_mode and self.filter_method is np.min:
                #     if tag_score[j] < self.filter_threshold:
                #         break  # has a token with confidence lower than threshold
                span_position = (i + offset, j + offset)  # with offset
                span_tup = span_position + (j - i + 1,)
                if span_position in tag_score:  
                    # include positive for training
                    span_score = tag_score[span_position]
                elif 'bin' in self.boundary_only_mode:
                    span_score = self.filter_method(
                        [tag_score[tok_idx] for tok_idx in range(i, j+1)])
                    if (self.filter_method in [np.min]) and tag_score[j] < self.filter_threshold:
                        break  # faster: all following spans contains this low-confidence token 
                elif 'bdy' in self.boundary_only_mode:
                    span_score = self.filter_method(
                        [tag_score[i][0], tag_score[j][1]])
                else:
                    raise ValueError(f"invalid boundary_only_mode: {self.boundary_only_mode}")
                # append it as a candidate span
                selected_span_score.append(span_score)
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                if (i, j) not in sent_ner:
                    sample['spans_label'].append(0)
                else:
                    sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                    if self.include_positive:  # for checking if there are not-recalled ones
                        del sent_ner[(i, j)]
                    num_cover += 1

        # extend spans not mentioned (only in training mode)
        if self.include_positive:
            for (i, j) in sent_ner:
                # longer width share the longest width embedding
                span_tup = (i + offset, j + offset,  
                            min(j - i + 1, self.max_span_length))
                selected_span_score.append(1.)
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                num_cover += 1

        # Pick out some of these spans 
        selected_indexes = self.distill_span_candidates(
            selected_span_score=selected_span_score,
            positive_span_count=len(positive_indexes))

        # the positive spans' score may be not high enough in argsort, we restore it.
        if self.include_positive:  
            for (_i, _j) in positive_indexes:
                if span2id[(_i, _j)] not in selected_indexes:
                    selected_indexes.append(span2id[(_i, _j)])
        sample['spans'] = [sample['spans'][_i] for _i in selected_indexes]
        sample['spans_label'] = [sample['spans_label'][_i] for _i in selected_indexes]
        num_cover = sum(map(lambda _sl: int(_sl > 0), sample['spans_label']))

        # in case of empty ones.
        if len(sample['spans']) == 0:  
            span_tup = (offset, offset, 1)
            sample['spans'].append(span_tup)
            sample['spans_label'].append(0)
        
        sample['positive_labels_count'] = len(sent_ner)
        sample['positive_labels_cover'] = num_cover
        assert len(sample['spans']) == len(sample['spans_label'])
        return sample

    def enumerating_with_span_score(self, sample, sent):
        # a list of [batch_size, input_dim]
        sent_ner = {}
        tag_score = {}
        positive_indexes = []

        text_len = len(sent.text)
        for ner in sent.ner:
            lef, rig = ner.span.span_sent
            sent_ner[(lef, rig)] = ner.label
            positive_indexes.append((lef, rig))
            if self.include_positive:
                tag_score[(lef, rig)] = 1.
        
        num_cover = 0
        sent_scores = sent.scores if sent.scores else []  
        offset = sample['sent_start']  # for context
        doc_offset = sample['sent_start_in_doc']  # for document
        for i, sent_score in enumerate(sent_scores):
            lef, rig, score = sent_score 
            # if 'span' in self.boundary_only_mode:
            cur_position = (lef - doc_offset + offset, rig - doc_offset + offset)
            tag_score[cur_position] = score

        span2id = {}
        sample['spans'] = []
        sample['spans_label'] = []
        selected_span_score = []
        
        # enumerate span candidates
        for i in range(text_len):
            for j in range(i, min(len(sent.text), i + self.max_span_length)):
                if self.drop_with_punc and sent.text[j] in self.PUNC_LIST:
                    break  # entity spans do not cover punctuations
                span_position = (i + offset, j + offset)  # with offset
                span_tup = span_position + (j - i + 1,)
                span_score = tag_score.get(span_position, 0.)
                # append it as a candidate span
                selected_span_score.append(span_score)
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                if (i, j) not in sent_ner:
                    sample['spans_label'].append(0)
                else:
                    sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                    if self.include_positive:  # for checking if there are not-recalled ones
                        del sent_ner[(i, j)]
                    num_cover += 1
        
        # extend spans not mentioned (only in training mode)
        if self.include_positive:
            for (i, j) in sent_ner:
                # longer width share the longest width embedding
                span_tup = (i + offset, j + offset,  
                            min(j - i + 1, self.max_span_length))
                selected_span_score.append(1.)
                sample['spans'].append(span_tup)
                span2id[(i, j)] = len(sample['spans']) - 1
                sample['spans_label'].append(self.ner_label2id[sent_ner[(i, j)]])
                num_cover += 1

        # Pick out some of these spans 
        selected_indexes = self.distill_span_candidates(
            selected_span_score=selected_span_score,
            positive_span_count=len(positive_indexes))

        # the positive spans' score may be not high enough in argsort, we restore it.
        if self.include_positive:  
            for (_i, _j) in positive_indexes:
                if span2id[(_i, _j)] not in selected_indexes:
                    selected_indexes.append(span2id[(_i, _j)])
        sample['spans'] = [sample['spans'][_i] for _i in selected_indexes]
        sample['spans_label'] = [sample['spans_label'][_i] for _i in selected_indexes]
        num_cover = sum(map(lambda _sl: int(_sl > 0), sample['spans_label']))

        # in case of empty ones.
        if len(sample['spans']) == 0:  
            span_tup = (offset, offset, 1)
            sample['spans'].append(span_tup)
            sample['spans_label'].append(0)

        sample['positive_labels_count'] = len(positive_indexes)
        sample['positive_labels_cover'] = num_cover
        assert len(sample['spans']) == len(sample['spans_label'])
        return sample

    def __call__(self, sample, sent, method=None, include_positive=None):
        if method:  # allow updating real-time
            self.method = method
        if include_positive is not None:
            self.include_positive = include_positive
        
        if self.boundary_only_mode in ['bin', 'bdy']:
            ret_sample = self.enumerating_one_char_spans(sample, sent)
        elif self.boundary_only_mode in ['bin_score', 'bdy_score']:
            # scores are given in samples.
            ret_sample = self.enumerating_with_tag_score(sample, sent)    
        elif self.boundary_only_mode in ['span_score']:
            # scores are given in samples.
            ret_sample = self.enumerating_with_span_score(sample, sent)    
        elif self.boundary_only_mode in ['span']:
            ret_sample = self.default_enumerating(sample, sent)    
        elif self.boundary_only_mode in ['targeted']:
            ret_sample = self.target_enumerating(sample, sent)    
        else:
            ret_sample = self.default_enumerating(sample, sent)
        return ret_sample
        

if __name__ == "__main__":
    # generate_inv_500_files()
    pass

