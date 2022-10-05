# coding: utf-8
# ==========================================================================
#   Copyright (C) 2016-2021 All rights reserved.
#
#   filename : pure_api.py
#   author   : chendian / okcd00@qq.com
#   date     : 2021-05-28
#   desc     : api class for loading and inference
# ==========================================================================
import time
import torch
import numpy as np
import json, jsonlines
from entity.models import EntityModel
from entity.utils import batchify, convert_dataset_to_samples, logger, NpEncoder
from shared.const import task_ner_labels, task_max_span_length, get_labelmap, CONFIG_FOR_PURE_API
from shared.data_structures import DictArgs, Document
from run_entity import get_label_mapping, get_span_filter, get_samples, evaluate
from modules.span_filter import SpanFilter


def np_softmax(x):
    e_x = np.exp(x.T - np.max(x, axis = -1))
    return (e_x / e_x.sum(axis=0)).T


class PureApi(object):
    # For inferring only.
    def __init__(self, args=None, load_model_dir=''):
        self.model = None
        self.digit_round = 8
        self.device = torch.device('cuda:0')

        # data containers
        self.js = None
        self.documents = None

        # configs, from ArgParses or DictArgs
        self.ner_label2id = None
        self.ner_id2label = None
        self.args = self.load_args(args)
        self.span_filter = None

        # load models
        if load_model_dir:
            self.init_models(self.args, load_model_dir)

    def dr(self, float_number, ignore_dr=True):
        # we keep the original confidence float value here.
        if ignore_dr:  
            return float_number
        
        # we can also round all the float numbers.
        if isinstance(float_number, list):
            return [self.dr(item, ignore_dr) for item in float_number]
        return round(float_number, self.digit_round)

    def load_args(self, args):
        default_args = CONFIG_FOR_PURE_API
        if args:
            default_args.update(args)  # args overwrite default settings
        self.args = DictArgs(default_args)
        if self.args.boundary_only_mode == 'none':
            label_case = task_ner_labels[self.args.task]
        elif 'score' in self.args.boundary_only_mode:  # predict phase
            label_case = task_ner_labels[self.args.task]
        else:
            label_case = task_ner_labels[self.args.boundary_only_mode]
        self.ner_label2id, self.ner_id2label = get_labelmap(label_case)
        return self.args

    def init_models(self, args, load_model_dir):
        # load from a directory with vocab, config and model.
        args.bert_model_dir = load_model_dir
        num_ner_labels = len(self.ner_label2id) + 1  # add 'O' tag
        self.model = EntityModel(args, num_ner_labels=num_ner_labels)

    def get_span_filter(self):
        if self.span_filter is None:
            self.span_filter = SpanFilter(
                ner_label2id=self.ner_label2id, 
                max_span_length=25,
                drop_with_punc=True,
                boundary_only_mode=self.args.boundary_only_mode,
                method=self.args.span_filter_method,
            )
        return self.span_filter

    def generate_document_from_sentences(self, sentences, save_path=None):
        # generate document obj without labels
        dict_list = [{  # in this case, there's only one doc
            "doc_key": 'default-doc',
            "sentences": [[c for c in text] for text in sentences],  # .lower()
            "ner": [[]] * len(sentences),
            "relations": [[]] * len(sentences)}]
        if save_path:
            self.save_as_jsonline(dict_list, save_path)
        self.js = dict_list
        self.documents = [Document(js) for js in self.js]
        return self.documents

    def save_as_jsonline(self, dict_list, save_path):
        # Save the documents as a jsonline file
        with jsonlines.open(save_path, mode='w') as writer:
            writer.write_all(dict_list)

    def load_from_jsonline(self, jsonl_file):
        self.js = [json.loads(line) for line in open(jsonl_file)]
        self.documents = [Document(js) for js in self.js]
        return self.documents

    def dump_prediction(self, save_path):
        # Save the prediction as a json file
        with open(save_path, 'w') as f:
            # turn numpy objects into python objects
            f.write('\n'.join(json.dumps(doc, cls=NpEncoder)
                              for doc in self.js))

    def turn_documents_into_batches(self, _data, span_filter):
        _samples, _ner = convert_dataset_to_samples(
            _data, self.args.max_span_length,
            context_window=self.args.context_window,
            span_filter=span_filter)
        _batches = batchify(
            _samples, self.args.eval_batch_size)
        print(f"documents have been transformed into batches with {_ner} entities.")
        return _batches

    def ner_predictions(self, batches, js=None, confidence_level=0.5, drop_non_entity=True):
        # span_hidden_table = {}
        ner_result = {}
        tot_pred_ett = 0
        ner_confidence = {}
        for i in range(len(batches)):  # 
            output_dict = self.model.run_batch(
                batches[i], training=False, 
                confidence_level=confidence_level)
            pred_ner = output_dict['pred_ner']
            pred_prob = output_dict['ner_probs']
            for sample, preds, probs in zip(batches[i], pred_ner, pred_prob):
                off = sample['sent_start_in_doc'] - sample['sent_start']
                k = f"{sample['doc_key']}-{str(sample['sentence_ix'])}"
                ner_result[k] = []
                ner_confidence[k] = []
                for span, pred, prob in zip(sample['spans'], preds, probs):
                    span_id = '%s::%d::(%d,%d)' % (
                        sample['doc_key'],
                        sample['sentence_ix'],
                        span[0] + off, span[1] + off)
                    # drop non-entity tuples from outputs.
                    if drop_non_entity and pred == 0:
                        continue
                    ner_result[k].append([
                        int(span[0] + off), 
                        int(span[1] + off),
                        self.ner_id2label.get(pred, 'O'),
                    ])
                    # confidence on each label
                    ner_confidence[k].append(
                        self.dr(np_softmax(prob).tolist()))
                tot_pred_ett += len(ner_result[k])

        logger.info('Total pred entities: %d' % tot_pred_ett)

        if js is None:
            js = self.js

        for i, doc in enumerate(js):
            doc["predicted_ner"] = []
            doc["predicted_prob"] = []
            doc["predicted_relations"] = []
            for j in range(len(doc["sentences"])):
                k = f"{doc['doc_key']}-{str(j)}"
                if k in ner_result:
                    doc["predicted_ner"].append(ner_result[k])
                    doc["predicted_prob"].append(ner_confidence[k])
                else:
                    logger.info('%s not in NER results!' % k)
                    doc["predicted_ner"].append([])
                    doc["predicted_prob"].append([])

                doc["predicted_relations"].append([])
            js[i] = doc
        return js

    def output_results(self, js=None):
        # consider about doc-level NER
        outputs = {}
        if js is None:
            js = self.js
        for doc in js:
            doc_id = doc['doc_key']
            results_in_doc = []
            ner = doc["predicted_ner"]
            prob = doc["predicted_prob"]
            sentences = doc['sentences']
            offset = 0
            for sent, ents in zip(sentences, ner):
                results_in_sent = []
                sent_len = len(sent)
                # sentence_text = ''.join(sent)
                for l, r, tp in ents:
                    results_in_sent.append(dict(
                        value=''.join(sent[l - offset: r - offset + 1]),
                        span=[int(l - offset), int(r - offset + 1)],
                        type=tp
                    ))
                results_in_doc.append(results_in_sent)
                offset += sent_len
            outputs[doc_id] = results_in_doc
        return outputs

    def output_results_as_samples_with_scores(self, js=None):
        # self.extract(documents=None, drop_non_entity=False)
        # set drop_non_entity in extract() as False, and then get the js for here
        if js is None:
            js = self.js

        candidate_counts = 0
        total_counts = 0
        for doc_idx, doc in enumerate(js):
            js[doc_idx]['scores'] = []  # same shape with sentences
            
            # doc_id = doc['doc_key']
            ner = doc["predicted_ner"]
            prob = doc["predicted_prob"]
            sentences = doc['sentences'] 
            # offset = 0
            for sent, ents, probs in zip(sentences, ner, prob):
                # for each sentence, its entities and corresponding probs
                sent_len = len(sent)
                # sentence_text = ''.join(sent)
                if self.args.boundary_only_mode == 'bin':
                    scores = [self.dr(_in) for _, _in in probs]
                elif self.args.boundary_only_mode == 'bdy':
                    scores = [self.dr([_begin+_single, _end+_single])
                              for _, _begin, _end, _single in probs]
                elif self.args.boundary_only_mode == 'span':
                    scores = [[int(ents[span_idx][0]), int(ents[span_idx][1]), self.dr(_prob)] 
                              for span_idx, (_, _prob) in enumerate(probs)]
                total_counts += len(scores)
                doc['scores'].append(scores)
                if self.args.boundary_only_mode == 'span':
                    scores = [item for item in scores if item[2] > 0.]
                candidate_counts += len(scores)
                # offset += sent_len
            js[doc_idx]['scores'] = doc['scores']  # necessary?
            js[doc_idx]['candidate_counts'] = [candidate_counts, total_counts]
            del js[doc_idx]['predicted_ner']  # no use for later
            del js[doc_idx]['predicted_prob']  # no use for later
        return js

    def add_scores_for_pure_samples(self, file_path, dump_path):
        # file_path = './data/resume/test.json'
        # dump_path = './data/resume/test.with_score.json'
        start_time = time.time()
        documents = self.load_from_jsonline(file_path)        
        load_time = time.time() - start_time
        
        js = self.extract(documents=documents, drop_non_entity=False)
        prediction_time = time.time() - start_time - load_time
        
        ret_samples = self.output_results_as_samples_with_scores(js=js)        
        self.save_as_jsonline(ret_samples, dump_path)
        dumping_time = time.time() - start_time - load_time - prediction_time

        print(f"Now saving pure samples with scores in {dump_path}")
        print("cost {:.2f}/{:.2f}/{:.2f} seconds for loading, prediction, dumping".format(
            load_time, prediction_time, dumping_time))
        print(ret_samples[0]['candidate_counts'])

    def output_results_for_p5(self, js):
        # no ``docs'' here, inputs are in a sentence text list.
        if js is None:
            js = self.js
        doc = js[0]
        results_in_doc = []
        prob = doc['predicted_prob']
        sentences, ner = doc['sentences'], doc['predicted_ner']
        offset = 0
        for sent, ents, ps in zip(sentences, ner, prob):
            results_in_sent = []
            sent_len = len(sent)
            # sentence_text = ''.join(sent)
            for e_idx, (l, r, tp) in enumerate(ents):
                ner_item = dict(
                    value=''.join(sent[l - offset: r - offset + 1]),
                    span=[int(l - offset), int(r - offset + 1)],
                    type=tp
                )
                if self.with_confidence:
                    ner_item.update({'prob': ps[e_idx]})
                results_in_sent.append(ner_item)
            results_in_doc.append(results_in_sent)
            offset += sent_len
        return results_in_doc

    def output_results_for_ccks(self, js):
        # no ``docs'' here, inputs are in a sentence text list.
        if js is None:
            js = self.js
        doc = js[0]
        results = []
        sentences, ner = doc['sentences'], doc['predicted_ner']
        offset = 0
        for s_idx, (sent, ents) in enumerate(zip(sentences, ner)):
            sent_len = len(sent)
            tags = ['O'] * sent_len
            # sentence_text = ''.join(sent)
            for l, r, tp in ents:
                for pivot in range(l - offset, r - offset + 1):
                    pos_t = 'I'
                    if pivot == l - offset:
                        pos_t = 'B'
                        if l == r:
                            pos_t = 'S'
                    elif pivot == r - offset:
                        pos_t = 'E'
                    tags[pivot] = '{}-{}'.format(pos_t, tp)
            results.append(u'\u0001'.join([
                str(s_idx + 1),
                ''.join(sent),
                ' '.join(tags)
            ]))
            offset += sent_len
        return results
    
    def evaluate(self, span_filter=None, documents=None):
        if span_filter is None:
            ner_label2id, ner_id2label = get_label_mapping(args=self.args)
            span_filter = get_span_filter(
                args=self.args, ner_label2id=ner_label2id, drop_with_punc=True)
        # predict on documents with labels, evaluate for performance
        span_filter.include_positive = False  # not training
        test_samples, test_ner = get_samples(
            data=documents, args=self.args, span_filter=span_filter)
        test_batches = batchify(
            test_samples, self.args.eval_batch_size)
        p, r, f1 = evaluate(self.model, test_batches, test_ner, output_prf=True)
        return p, r, f1

    def extract(self, documents=None, drop_non_entity=True, span_filter=None):
        if documents is None:
            # or load documents with
            # documents = self.load_from_jsonline('./test.json')
            documents = self.documents
        if span_filter is None:
            span_filter = self.get_span_filter()
        test_batches = self.turn_documents_into_batches(
            _data=documents, span_filter=span_filter)
        self.js = self.ner_predictions(
            batches=test_batches, js=self.js, 
            drop_non_entity=drop_non_entity)
        return self.js

    def batch_extract(self, sentences, output_method='p5', with_confidence=False, drop_non_entity=True):
        self.with_confidence = with_confidence
        documents = self.generate_document_from_sentences(sentences)
        answers = self.extract(documents, drop_non_entity=drop_non_entity)
        if output_method in ['ccks']:
            return self.output_results_for_ccks(answers)
        if output_method in ['p5']:
            return self.output_results_for_p5(answers)
        return self.output_results(answers)


def test_10000_cases():
    pure_api = PureApi(
        load_model_dir='/home/chendian/PURE/output_dir/findoc_old/')

    # input:  ['今天我在庖丁科技有限公司吃饭。',
    #          '拼多多这个公司真是太拼了。']
    documents = pure_api.load_from_jsonline(
        jsonl_file='/home/chendian/PURE/data/findoc/test.json')

    result_js = pure_api.extract(documents=documents)

    # output: [[{'span': [4, 12], 'value': '庖丁科技有限公司', 'type': 'company'}],
    #          [{'span': [0, 3], 'value': '拼多多', 'type': 'company'}]]
    answers = pure_api.output_results_for_p5(js=result_js)
    return answers


if __name__ == "__main__":
    pa = PureApi(args=None, load_model_dir='/home/chendian/PURE/output_dir/findoc_old/')
    answers = pa.batch_extract(['庖丁科技是一家金融科技公司', '国务院发布了新的《行政管理办法》'])
