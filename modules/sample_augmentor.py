# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : sample_augmentor.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-11-01
#   desc     :
# ==========================================================================
import os
import copy
import json
import torch
import pickle
import random
import numpy as np
import jsonlines
from tqdm import tqdm
from entity.utils import (
    batchify, convert_dataset_to_samples,
    logger, output_ner_predictions, NpEncoder)


def flatten(nested_list, unique=False):
    ret = [elem for sub_list in nested_list for elem in sub_list]
    if unique:
        return list(set(ret))
    return ret


class SampleAugmentor(object):
    def __init__(self, sample_file_dir, elib_path=None, task_name=None, augment_counts=-1):
        super().__init__()
        # 2 ways to determine whether to augment on current sample.
        self.augment_rate = 0.25
        self.augment_counts = augment_counts

        self.chars_in_entity = []
        self.entity_library = self.load_elib_file(
            sample_file_dir=sample_file_dir, 
            elib_path=elib_path)

        from shared.const import get_labelmap, task_ner_labels
        self.ner_label2id, self.ner_id2label = None, None
        if task_name:
            self.ner_label2id, self.ner_id2label = get_labelmap(
                task_ner_labels[task_name])

    def load_elib_file(self, sample_file_dir, elib_path=None):
        if elib_path is None:
            elib_path = sample_file_dir + '/train.elib'
        if os.path.exists(elib_path):
            elib_dict = pickle.load(open(elib_path, 'rb'))
        else:
            elib_dict = self.generate_elib_file(
                sample_file_dir=sample_file_dir)
        chars_in_entity = set()
        for ent_typ in elib_dict:
            for ent_length in elib_dict[ent_typ]:
                chars_in_entity |= set(flatten(elib_dict[ent_typ][ent_length]))
        self.chars_in_entity = list(chars_in_entity)
        print(f"The entities in current dataset have {len(chars_in_entity)} unique chars.")
        return elib_dict

    def generate_elib_file(self, sample_file_dir, elib_path=None):
        elib_dict = {}
        train_path = sample_file_dir + '/train.json'
        offset = 0
        with jsonlines.open(train_path, 'r') as reader:
            for doc in tqdm(reader):
                sentences = doc["sentences"]
                for idx, sent in enumerate(doc["ner"]):
                    sentence = sentences[idx]
                    for lef, rig, tag in sent:
                        elib_dict.setdefault(tag, {})
                        # lef == rig for single-token word (lef, rig) in PURE
                        text = ''.join(sentence[lef-offset: rig-offset+1])
                        text_len = rig - lef + 1
                        elib_dict[tag].setdefault(text_len, set())
                        elib_dict[tag][text_len].add(text)
                    offset += len(sentence)
        if elib_path is None:
            elib_path = sample_file_dir + '/train.elib'
        pickle.dump(elib_dict, open(elib_path, 'wb'))
        return elib_dict  

    def get_samples_from_jsonl(self, jsonl_path, ner_label2id, is_training=False):
        from shared.data_structures import Dataset
        data = Dataset(jsonl_path)
        
        import numpy as np
        span_filter = SpanFilter(
            ner_label2id=self.ner_label2id or ner_label2id,
            max_span_length=25,
            drop_with_punc=True,
            filter_method=np.max,
            filter_threshold=0.0,
            boundary_only_mode='targeted',
            method='',
        )

        _samples, _ner = convert_dataset_to_samples(
            data, 25, context_window=0,
            span_filter=span_filter, 
            is_training=False)
        return _samples

    def get_jsonl_from_samples(self, samples, output_file=None, doc_key='dummy'):
        js = []
        doc = {'sentences': [], 'ner': [], 'relations': [], 'doc_key': doc_key}
        offset = 0
        for sample in tqdm(samples):
            doc['sentences'].append(sample['tokens'])
            ner_case = []
            for _sp, _l in zip(sample['spans'], sample['spans_label']):
                ner_case.append([_sp[0], _sp[1], self.ner_id2label[_l]])
            doc['ner'].append(ner_case)
            offset += len(sample['tokens'])
        doc['relations'] = [[] for _ in doc['ner']]
        js.append(doc)
        if output_file:
            with open(output_file, 'w') as f:
                f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))
        return js

    def sample_from_jsonl(self, jsonl_path, n_samples=500):
        with jsonlines.open(jsonl_path, 'r') as reader:
            for doc in tqdm(reader):
                # sample in the first doc (no multi-doc datasets in CNER)
                selected_indexes = []
                sentences = doc["sentences"]
                ner_case = doc["ner"]
                target_ner_position = []

                offset = 0
                for idx, sent in enumerate(ner_case):
                    sent = [item for item in sent if item[2] in ["人名", "公司"]]
                    if len(sent) == 1:
                        selected_indexes.append(idx)
                        target_ner_position.append([sent[0][0]-offset, sent[0][1]-offset, sent[0][2]])
                    else:
                        target_ner_position.append([])
                    offset += len(sentences[idx])

                print('1-ent samples: ', len(selected_indexes))
                selected_indexes = random.sample(selected_indexes, 
                                                 k=min(n_samples, len(selected_indexes)))
                selected_ner_pos = [target_ner_position[idx] for idx in selected_indexes]

                new_doc = {
                    "doc_key": f"{doc['doc_key']}-selected-500",
                    'sentences': [sentences[idx] for idx in selected_indexes],
                    'ner': [],
                    'relations': [[] for _ in selected_indexes],
                }
                offset = 0
                # print(len(new_doc['sentences']), len(selected_ner_pos))
                for idx, (lef, rig, tag) in enumerate(selected_ner_pos):
                    ner_case = []
                    ner_case.append([lef+offset, rig+offset, tag])
                    new_doc['ner'].append(ner_case)
                    offset += len(new_doc['sentences'][idx])
                return new_doc
        return None

    def gather_entities_from_samples(self, samples):
        ent_case = []
        for sample in samples:
            tokens = sample['tokens']
            for lef, rig, _ in sample['spans']:
                ent = tokens[lef:rig+1]
                if ent not in ent_case:
                    ent_case.append(ent_case)
        return ent_case

    def generate_inv_samples(self, samples, entities=None):
        # here samples and entities should only contain one ent_type
        inv_samples = []

        # here each sample has keys (not the samples in jsonl_file)
        # ['tokens', 'spans', 'spans_label', 'sent_start', 'sent_end', 'sent_length']
        # ['positive_labels_count', 'positive_labels_cover']
        if entities is None:
            entities = self.gather_entities_from_samples(samples)
        for sample_idx, sample in enumerate(samples):
            lef, rig, _ = sample['spans'][0]
            for ent_idx, ent_text in enumerate(entities):
                new_sample = self.change_tokens_in_sample(
                    copy.deepcopy(sample), lef, rig, ent_text)
                inv_samples.append(new_sample)
        return inv_samples

    def another_entity_with_same_type(self, ent_text, same_length=False):
        if same_length:
            ent_len = len(ent_text)
        candidates = []
        for typ in self.entity_library:
            if same_length:
                if ent_len not in self.entity_library[typ]:
                    continue
                if ent_text in self.entity_library[typ][ent_len]:
                    candidates.extend(list(self.entity_library[typ][ent_len]))
            else:
                for ent_len in self.entity_library[typ]:
                    candidates.extend(list(self.entity_library[typ][ent_len]))
        if len(candidates) == 0:
            raise ValueError(f"Invalid ent_text: {ent_text}")
        return [_c for _c in random.choice(candidates)]

    def another_entity_with_any_type(self):
        candidates = []
        for typ in self.entity_library:
            for ent_len in self.entity_library[typ]:
                candidates.extend(list(self.entity_library[typ][ent_len]))
        if len(candidates) == 0:
            raise ValueError(f"Implement Error in Any-Type generation")
        return [_c for _c in random.choice(candidates)]

    def generate_random_entity_name(self, ent_length=None, fn=random.choices):
        # fn: [random.choices|random.sample]
        candidates = self.chars_in_entity
        if ent_length is None:    
            ent_length = random.choice([2,3,4])
        ignores = flatten([self.entity_library[k][ent_length] 
                           for k in self.entity_library if ent_length in self.entity_library[k]])
        selected_chars = fn(list(set(candidates) - set(ignores)), k=ent_length)
        # print(selected_chars)
        return selected_chars

    def change_tokens_in_sample(self, sample, lef, rig, changed_text):
        offset_delta = len(changed_text) - (rig - lef + 1)
        if offset_delta + len(sample['tokens']) > 500:
            return sample  # in case of longer than 512 (error in BERT)
        sample['tokens'] = sample['tokens'][:lef] + changed_text + sample['tokens'][rig+1:]
        if offset_delta != 0: 
            sample = self.shift_spans_in_sample(sample, lef, rig, offset_delta)
        sample['sent_end'] += offset_delta 
        sample['sent_length'] += offset_delta 
        assert len(sample['tokens']) == sample['sent_length']
        return sample

    def shift_spans_in_sample(self, sample, lef, rig, offset_delta):
        _end = len(sample['tokens']) - 1

        def bound(_v, _lb, _rb=_end):
            _v = min(_rb, max(_lb, _v))
            _v = max(0, _v)
            return _v

        for i, sp in enumerate(sample['spans']):
            if sp[1] <= lef:  # XXXOLR
                continue

            _sp0, _sp1 = -1, -1
            if rig < lef:  # Insertion
                _sp0 = sp[0] if sp[0] < lef else bound(sp[0] + offset_delta, lef)
                _sp1 = sp[1] if sp[1] < lef else bound(sp[1] + offset_delta, _sp0)
                new_span = (_sp0, _sp1, _sp1 - _sp0 + 1)
            elif sp[0] >= rig:  # LROXXX
                new_span = (
                    sp[0] + offset_delta, 
                    sp[1] + offset_delta, 
                    sp[2])
            elif lef < sp[0] < rig: 
                _sp0 = bound(sp[0] + offset_delta, lef)
                if sp[1] >= rig:  # LOXXRX
                    _sp1 = bound(sp[1] + offset_delta, lef + sp[2] - 1 + offset_delta)
                else:  # sp[1] < rig:  # LOXXOR
                    _sp1 = _sp1 = bound(sp[1] + offset_delta, _sp0)
                _sp1 = bound(_sp1, _sp0)
                new_span = (_sp0, _sp1, _sp1 - _sp0 + 1)
            elif sp[0] == lef:  # LXXX / XXXR
                _sp0 = sp[0]
                _sp1 = bound(sp[1] + offset_delta, _sp0)
                new_span = (_sp0, _sp1, _sp1 - _sp0 + 1)
            elif sp[0] < lef:
                if lef <= rig <= sp[1]:  # XLXXRX
                    _sp0 = sp[0]
                    _sp1 = bound(sp[1] + offset_delta, _sp0)
                    new_span = (_sp0, _sp1, _sp1 - _sp0 + 1)
                elif lef <= sp[1] < rig:  # XLXXOR
                    _sp0 = sp[0]
                    _sp1 = bound(sp[1] + offset_delta, _sp0)
                    new_span = (_sp0, _sp1, _sp1 - _sp0 + 1)
                else:  # sp[1] < lef:  # XXXOLR
                    new_span = (sp[0], sp[1], sp[1]-sp[0]+1)
            else:
                raise ValueError(f"Invalid sequence: L:{lef}, R:{rig}, span:({sp[0]}, {sp[1]})")
            
            try:
                assert _sp0 <= _sp1
            except:
                print(sample['tokens'])
                print(f"Invalid sequence: L:{lef}, R:{rig}, span:({sp[0]}, {sp[1]})")
                print(_sp0, _sp1)
                print(sample['spans'][i])
                raise ValueError()
            
            sample['spans'][i] = new_span
        # drop invalid spans
        # dropped_indexes = [i for i, item in enumerate(sample['spans']) if item[1] >= item[0]]
        # sample['spans'] = [item for i, item in enumerate(sample['spans']) if i in dropped_indexes]
        # sample['spans_label'] = [item for i, item in enumerate(sample['spans_label']) if i in dropped_indexes]
        return sample

    def boundary_movement(self, sample, target_span_index):
        # positive, X X X -
        rn = random.random()
        changed_flag = True
        _l, _r, _width = sample['spans'][target_span_index]
        if 0. <= rn < 0.25 and 0 < _l:
            _l -= 1
        elif 0.25 <= rn < 0.5 and _l < _r:
            _l += 1
        elif 0.5 <= rn < 0.75 and _l < _r:
            _r -= 1
        elif _r < len(sample['tokens']) - 1:
            _r += 1
        else:
            changed_flag = False
        
        if changed_flag:
            sample['spans'][target_span_index] = (_l, _r, _width)
            sample['spans_label'][target_span_index] = 0
        return sample

    def mention_replacement(self, sample, target_span_index):
        # positive, V V V -
        target_span = sample['spans'][target_span_index]
        offset = sample['sent_start']
        _l, _r = target_span[0] - offset, target_span[1] - offset

        # get another entity with the same length
        target_text = sample['tokens'][_l: _r+1]
        changed_text = self.another_entity_with_same_type(target_text)
        sample = self.change_tokens_in_sample(sample, _l, _r, changed_text)        
        return sample

    def random_replacement(self, sample, target_span_index):
        # positive, X V - CTX
        target_span = sample['spans'][target_span_index]
        offset = sample['sent_start']
        _l, _r = target_span[0] - offset, target_span[1] - offset
        target_text = sample['tokens'][_l: _r+1]
        random_text = self.generate_random_entity_name(len(target_text))
        sample = self.change_tokens_in_sample(sample, _l, _r, random_text)

        sample['spans_label'][target_span_index] = -1  # ignore
        sample['spans_alpha_label'][target_span_index] = 0  # context
        sample['positive_labels_count'] -= 1
        sample['positive_labels_cover'] -= 1
        return sample

    def mention_insertion(self, sample, target_span_index):
        # negative, V X - NAME
        target_span = sample['spans'][target_span_index]
        offset = sample['sent_start']
        _l = target_span[0] - offset  # the position for insertion
        changed_text = self.another_entity_with_any_type()
        sample = self.change_tokens_in_sample(sample, _l, _l-1, changed_text)                
        sample['spans_label'][target_span_index] = -1  # ignore
        sample['spans_alpha_label'][target_span_index] = 1  # name
        return sample

    def random_insertion(self, sample, target_span_index):
        # negative, X X - -
        target_span = sample['spans'][target_span_index]
        offset = sample['sent_start']
        _l = target_span[0] - offset  # the position for insertion
        random_text = self.generate_random_entity_name()
        sample = self.change_tokens_in_sample(sample, _l, _l-1, random_text)
        return sample

    def augment(self, sample, method='random'):
        sample['spans_alpha_label'] = [
            -1 for _ in sample['spans_label']]
        positive_positions = [
            i for i, _l in enumerate(sample['spans_label']) if _l > 0]
        
        if len(positive_positions) > 0:
            if method == 'random':
                method = random.choice(['mr', 'rr'])
            # augment on 25% samples
            if random.random() > self.augment_rate:
                return None  
            target_span_index = random.choice(
                positive_positions)
        else:
            if method == 'random':
                method = random.choice(['mi', 'ri'])
            # add this: balance label counts
            target_span_index = random.choice(
                list(range(len(sample['spans']))))

        func_dict = {
            'mr': self.mention_replacement,
            'rr': self.random_replacement,
            'bm': self.boundary_movement,
            'mi': self.mention_insertion,
            'ri': self.random_insertion,
            'origin': lambda x: x,
        }
        # if method in ['mr', 'rr']:  # ablation: remove insertion
        # if method in ['mi', 'ri']:  # ablation: remove replacement
        sample = func_dict.get(method)(sample, target_span_index)
        return sample

    def __call__(self, sample, method='random'):
        # each sample has keys 
        # ['tokens', 'spans', 'spans_label', 'sent_start', 'sent_end', 'sent_length']
        # ['positive_labels_count', 'positive_labels_cover']
        if isinstance(sample, list):
            if self.augment_counts > 0:
                positive_samples = [_s for _s in sample if any([_l for _l in _s['spans_label'] if _l > 0])]
                negative_samples = [_s for _s in sample if not any([_l for _l in _s['spans_label'] if _l > 0])]
                positive_samples = random.sample(positive_samples, k=self.augment_counts)
                negative_samples = random.sample(negative_samples, k=self.augment_counts)
                new_samples = positive_samples + negative_samples
                random.shuffle(new_samples)
                self.augment_rate = 1.01  # augment on all selected samples
            ret = [item for item in list(map(
                lambda _s: self(_s, method=method), 
                new_samples)) if item]
            print("# Augmented Samples:", len(ret))
            return ret
        _sample = copy.deepcopy(sample)
        return self.augment(_sample, method=method)


def generate_inv_500_files():
    task_name = 'msra_origin'
    augmentor = SampleAugmentor(
        sample_file_dir=f'/home/chendian/PURE/data/{task_name}/', 
        elib_path=None)
    test_json_path = f'/home/chendian/PURE/data/{task_name}/test.json'
    new_doc = augmentor.sample_from_jsonl(test_json_path)
    inv_dump_path = f'/home/chendian/PURE/data/{task_name}/inv_test_500.json'
    with jsonlines.open(inv_dump_path, 'w') as writer:
        writer.write(new_doc)


def generate_inv_heatmap_files():
    # view generate_inv_samples.ipynb
    pass

if __name__ == "__main__":
    generate_inv_500_files()
