import os
import json
import time
import torch
import logging
import numpy as np


logger = logging.getLogger('root')


def batchify(samples, batch_size, show_single_batch=False):
    """
    Batchfy samples with a batch size
    """
    num_samples = len(samples)
    list_samples_batches = []

    # if a sentence is too long, make itself a batch to avoid GPU OOM
    to_single_batch = []
    for i in range(0, len(samples)):
        if len(samples[i]['tokens']) > 350:
            to_single_batch.append(i)

    for i in to_single_batch[::-1]:
        try:
            if show_single_batch:
                logger.info('Single batch sample: %s-%d',
                            samples[i]['doc_key'],
                            samples[i]['sentence_ix'])
        except Exception as e:
            print(samples.__len__(), i)
            print(e)
            continue
        list_samples_batches.append([samples[i]])
    samples = [sample for i, sample in enumerate(samples) if i not in to_single_batch]

    for i in range(0, len(samples), batch_size):
        list_samples_batches.append(samples[i:i + batch_size])

    assert (sum([len(batch) for batch in list_samples_batches]) == num_samples)
    return list_samples_batches


def overlap(s1, s2):
    if s2.start_sent >= s1.start_sent and s2.start_sent <= s1.end_sent:
        return True
    if s2.end_sent >= s1.start_sent and s2.end_sent <= s1.end_sent:
        return True
    return False

def tokens_adaption(tokens):
    
    def _adapt(_token):
        if _token in '“”':  # [UNK]
            return '"'  # 107
        if _token in "‘’":  # [UNK]
            return "'"  # 112
        if _token in "—":  # [UNK]
            return "-"  # 118
        return _token

    return list(map(_adapt, tokens))

def convert_dataset_to_samples(dataset, max_span_length, 
                               context_window=0, split=0, 
                               span_filter=None, is_training=None):
    """
    Extract sentences and gold entities from a dataset
    """
    # split: split the data into train and dev (for ACE04)
    # split == 0: don't split
    # split == 1: return first 90% (train)
    # split == 2: return last 10% (dev)
    samples = []
    num_ner = 0
    max_len = 0
    max_ner = 0
    num_cover = 0  # how many truth entities are covered by span candidates
    num_overlap = 0  # span overlap
    num_candidates = 0  # how many span candidates we generated

    if split == 0:
        data_range = (0, len(dataset))
    elif split == 1:
        data_range = (0, int(len(dataset) * 0.9))
    elif split == 2:
        data_range = (int(len(dataset) * 0.9), len(dataset))

    for c, doc in enumerate(dataset):
        if c < data_range[0] or c >= data_range[1]:
            continue
        for i, sent in enumerate(doc):
            sample = {
                'doc_key': doc._doc_key,
                'sentence_ix': sent.sentence_ix,
            }
            if context_window != 0 and len(sent.text) > context_window:
                logger.info('Long sentence: {} {}'.format(sample, len(sent.text)))
                # print('Exclude:', sample)
                # continue
            sample['tokens'] = tokens_adaption(sent.text)
            assert len(sample['tokens']) == len(sent.text)
            sample['sent_length'] = len(sent.text)
            sent_start = 0
            sent_end = len(sample['tokens'])

            max_len = max(max_len, len(sent.text))
            max_ner = max(max_ner, len(sent.ner))

            if context_window > 0:
                add_left = (context_window - len(sent.text)) // 2
                add_right = (context_window - len(sent.text)) - add_left

                # add left context
                j = i - 1
                while j >= 0 and add_left > 0:
                    context_to_add = doc[j].text[-add_left:]
                    sample['tokens'] = context_to_add + sample['tokens']
                    add_left -= len(context_to_add)
                    sent_start += len(context_to_add)
                    sent_end += len(context_to_add)
                    j -= 1

                # add right context
                j = i + 1
                while j < len(doc) and add_right > 0:
                    context_to_add = doc[j].text[:add_right]
                    sample['tokens'] = sample['tokens'] + context_to_add
                    add_right -= len(context_to_add)
                    j += 1

            sample['sent_start'] = sent_start
            sample['sent_end'] = sent_end
            sample['sent_start_in_doc'] = sent.sentence_start

            # if is training phase, extend all positive spans.
            sample = span_filter(sample, sent, include_positive=is_training)
            num_ner += sample.get('positive_labels_count')
            num_cover += sample.get('positive_labels_cover')
            num_candidates += len(sample['spans_label'])
            samples.append(sample)

    avg_length = sum([len(sample['tokens']) for sample in samples]) / len(samples)
    max_length = max([len(sample['tokens']) for sample in samples])
    logger.info('# Overlap: %d' % num_overlap)
    logger.info('Extracted %d samples from %d documents, with %d NER labels, %.3f avg input length, %d max length' % (
        len(samples), data_range[1] - data_range[0], num_ner, avg_length, max_length))
    logger.info('Max Length: %d, max NER: %d' % (max_len, max_ner))
    logger.info('Span Candidates\' Count: %d, Cover: %d' % (num_candidates, num_cover))
    return samples, num_ner


def ner_predictions(model, batches, dataset, ner_id2label, skip_non_entity=True):
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    torch.cuda.empty_cache()
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        pred_probs = output_dict['ner_probs']
        for sample_idx, (sample, preds) in enumerate(zip(batches[i], pred_ner)):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = f"{str(sample['doc_key'])}-{str(sample['sentence_ix'])}"
            ner_result[k] = []
            prob_mat = pred_probs[sample_idx]
            for span_idx, (span, pred) in enumerate(zip(sample['spans'], preds)):
                # print(span)
                prob = prob_mat[span_idx]
                span_id = '%s::%d::(%d,%d)' % (
                    sample['doc_key'], sample['sentence_ix'], 
                    span[0] + off, span[1] + off)
                if pred == 0 and skip_non_entity:
                    continue  # remove most spans with the not-an-entity label.
                res = [span[0] + off, span[1] + off, ner_id2label[pred]]
                # print(off, span, res)
                ner_result[k].append(res)
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = f"{str(doc['doc_key'])}-{str(j)}"
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])
            doc["predicted_relations"].append([])
        js[i] = doc
    torch.cuda.empty_cache()
    return js


def output_ner_predictions(model, batches, dataset, output_file, ner_id2label):

    # predict ner and save results in js
    js = ner_predictions(model, batches, dataset, ner_id2label)
    
    # Save the prediction as a json file
    logger.info('Output predictions to %s..'%(output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def get_train_fold(data, fold):
    print('Getting train fold %d...' % fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i < l or i >= r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d' % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def get_test_fold(data, fold):
    print('Getting test fold %d...' % fold)
    l = int(len(data) * 0.1 * fold)
    r = int(len(data) * 0.1 * (fold + 1))
    new_js = []
    new_docs = []
    for i in range(len(data)):
        if i >= l and i < r:
            new_js.append(data.js[i])
            new_docs.append(data.documents[i])
    print('# documents: %d --> %d' % (len(data), len(new_docs)))
    data.js = new_js
    data.documents = new_docs
    return data


def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)


def dump_json(obj, fp):
    try:
        fp = os.path.abspath(fp)
        if not os.path.exists(os.path.dirname(fp)):
            os.makedirs(os.path.dirname(fp))
        with open(fp, 'w', encoding='utf8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=4, separators=(',', ':'))
        print(f'json文件保存成功，{fp}')
        return True
    except Exception as e:
        print(f'json文件{obj}保存失败, {e}')
        return False
