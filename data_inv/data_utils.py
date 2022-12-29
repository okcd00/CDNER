# coding: utf-8
# ==========================================================================
#   Copyright (C) 2020 All rights reserved.
#
#   filename : data_utils.py
#   author   : chendian / okcd00@qq.com
#   date     : 2020-08-21
#   desc     : data utils for data pre-processing and post-processing
#              mainly used in api.py
# ==========================================================================
import re, copy, tqdm, logging
from tools.encodings import *
from modules.database import Database
from anywhere import get_path


TAIL_NAMES = ['name_aware', 'context_aware']
TARGET_CLASSES = [u"", u"公司", u"人名", u"地址", u"产品业务", u"文件"]
TARGET_CLASSES_EN = [u"", u"COM", u"PER", u"LOC", u"PROD", u"MISC"]

TARGET_CLASSES_MSRA = [u"", u"公司", u"人名", u"地址"]
TARGET_CLASSES_ONTO4 = [u"", u"ORG", u"GPE", u"PER", u"LOC"]
TARGET_CLASSES_RESUME = [u"", u"公司", u"人名", u"地址", u"学历", u"专业", u"国籍", u"民族", u"职称"]
TARGET_CLASSES_CCKS = [
    'assist', 'cellno', 'city', 'community', 'devzone', 'distance', 'district',
    'floorno', 'houseno', 'intersection', 'poi', 'prov', 'road', 'roadno',
    'subpoi', 'town', 'village_group']

LS_AUXILIARY = {
    'DocNER': [
        'O',
        'B-公司', 'B-人名', 'B-地址', 'B-产品业务', 'B-文件',
        'I-公司', 'I-人名', 'I-地址', 'I-产品业务', 'I-文件',
        'E-公司', 'E-人名', 'E-地址', 'E-产品业务', 'E-文件',
    ],
    'SentNER': [
        'O',
        'B-公司', 'B-人名', 'B-地址', 'B-产品业务', 'B-文件',
        'I-公司', 'I-人名', 'I-地址', 'I-产品业务', 'I-文件',
        'E-公司', 'E-人名', 'E-地址', 'E-产品业务', 'E-文件',
    ]
}
DOC_MAP = json_load('./data/document_ids.json')
VOCAB_LIST = [line.strip() for line in open(get_path('bert_vocab'))]
SINGLE_CHAR_VOCAB_LIST = [c for c in VOCAB_LIST if c.__len__() == 1]


def load_database(db_path, read_only=True):
    return Database(db_path, read_only=read_only)


def dump_database(data, db_path):
    db = Database(db_path, read_only=False)
    db.write(data)
    return db


def save_prepared_cc_data(path, data):
    """
    将生成的一致性模型数据存储下来
    :param path: 存放一致性模型数据的路径
    :param data: 需要存下来的一致性模型数据，通常为 data_by_doc_id
    :return:
    """
    data_index = 0
    for item in data:
        samples = data['sentence']
        doc_id = get_document_id(samples[0])
        if 'info' not in item:
            item['info'] = {}
        item['info'].update(
            {'sid': '{}_{}'.format(doc_id, data_index)})
        data_index += 1
    dump_database(data=data, db_path=path)


def get_position_from_entity(entity, words=None):
    """
    返回当前 entity 在 words 当中的 indexes 列表（注意不是id列表，id可能不连续）
    :param entity:
    :param words:
    :return:
    """
    if words is None:
        words = entity['words']
    id2index = {w['id']: w_idx for w_idx, w in enumerate(words)}
    position = sorted(list(map(id2index.get, entity['tokens'])))
    return position


def get_position_tuple_from_entity(ent, words):
    ent_pos = get_position_from_entity(ent, words)
    ent_span = tuple([ent_pos[0], ent_pos[-1] + 1])
    return ent_span


def get_string_from_entity(entity, words=None):
    position = get_position_from_entity(entity=entity, words=words)
    ent_string = ''.join([words[p]['word'] for p in position])
    return ent_string


def is_table_sample(sample, sid=None):
    # only used to judge for sl_samples
    if sample and not isinstance(sample, dict):
        return False
    if sid is None:  # get sid from sample
        sid = sample['info']['sid']
    if re.match('^[0-9]+(row|col)[0-9]+$', sid.split('_')[-1]):
        return True
    return False


def is_text_sample(sample, sid=None):
    # only used to judge for sl_samples
    if sample and not isinstance(sample, dict):
        return False
    if sid is None:  # get sid from sample
        sid = sample['info']['sid']
    if re.match('^[0-9]+_(output|generated)(-[0-9]+){1,3}_?$', sid):
        return True
    if re.match('^[0-9]+-.+(-[0-9]+){1,3}$', sid):
        return True
    if re.match('^[0-9]+_[0-9]+$', sid):
        return True
    return False


def add_text_key_in_sample(sample):
    for entity in sample["entities"]:
        if entity['type'] == u"项目":
            entity['type'] = u"产品业务"
    sentence = u""
    for i in sample["words"]:
        if len(i["word"]) != 1:
            # sentence += '\n'
            if i['word'].lower() == '[unused10]':
                sentence += '\x01'
            elif i['word'].lower() == '[unused11]':
                sentence += '\x02'
            else:  # others
                sentence += '\x03'
        else:
            sentence += i["word"]
    sample["text"] = sentence
    return sample


def add_span_in_sample(sample):
    _sample, _old2new = re_id_func(copy.deepcopy(sample))
    word_ids = [w['id'] for w in _sample['words']]
    _entities = []
    for ent_idx, ent in enumerate(_sample['entities']):
        try:
            span_list = sorted([word_ids.index(t) for t in ent['tokens']])
        except Exception as e:
            print(str(e))
            print("the tokens {} is not in words\nwords:{}".format(
                ', '.join(ent['tokens']), ', '.join(word_ids)))
            continue
        try:
            span_tuple = (span_list[0], span_list[-1] + 1)
            sample['entities'][ent_idx]['span'] = span_tuple
            _entities.append(sample['entities'][ent_idx])
        except Exception as e:
            raise ValueError(str(e) + ' with {}\n{}'.format(ent, _sample))
    sample['entities'] = _entities
    return sample


def add_mask_for_context_inputs(inputs, position):
    tokens = inputs['tokens']
    mask_idx = VOCAB_LIST.index('[MASK]')
    for idx, (lef, rig) in enumerate(position):
        tokens[lef:rig, idx] = mask_idx
    inputs['tokens'] = tokens
    return inputs


def remove_empty_entities_in_sample(sample):
    sample['entities'] = [ent for ent in sample['entities'] if len(ent['tokens']) > 0]
    return sample


def reid_entities(label_result, word_old2new):

    def create_entity_id(operands, type_name):
        if isinstance(operands, list) or isinstance(operands, tuple):
            return u'({}:{})'.format(type_name, ','.join(map(str, operands)))
        elif isinstance(operands, str):
            return u'({}:{})'.format(type_name, operands)
        else:
            raise TypeError(type(operands))
        # return create_id(operands, type_name)

    def reid_entity(_entity):
        if isinstance(_entity['tokens'], tuple) or isinstance(_entity['tokens'], list):
            _entity['tokens'] = [word_old2new[t] for t in _entity['tokens']]
        else:
            _entity['tokens'] = word_old2new[_entity['tokens']]

        _new_id = create_entity_id(_entity['tokens'], _entity['type'])
        entity_old2new[_entity['id']] = _new_id
        _entity['id'] = _new_id
        if 'sub_entities' in _entity:
            for e in _entity['sub_entities']:
                reid_entity(e)

    entity_old2new = {}
    for entity in label_result['entities']:
        reid_entity(entity)
    return label_result, entity_old2new


def re_id_func(label_result, readable=True, text_only=False):
    # reid_words_entities
    # label_result['info']['sid'] = unicode(label_result['info']['sid'])
    if readable or text_only:
        word_old2new = {}
        for i, word in enumerate(label_result['words']):
            if text_only:
                new_id = word['word']
            else:
                new_id = u'w{}|{}'.format(i, word['word'])
            word_old2new[word['id']] = new_id
            word['id'] = new_id
        _, old2new = reid_entities(label_result, word_old2new)

        # assert 'pa_tokens' in label_result
        # this is a temp fix for 'pa_tokens' in usage_example/is_a/match
        if 'pa_tokens' in label_result:
            for i, token_id in enumerate(label_result['pa_tokens']):
                label_result['pa_tokens'][i] = old2new[token_id]
    else:
        old2new = {entity['id']: entity['id'] for entity in label_result['entities']}
    return label_result, old2new


def get_file_id(sample=None, sid=None, data_type=None):
    """
    获得 file_id，通常是 sid 的子串
    :param sample: 传入一个 UTIE-data 的 dict (sample)
    :param sid: 传入 sid (in sample)
    :param data_type: 传入当前 sample/sid 所属的数据类型 (in sid)
    :return: file_id
    """
    if sid is None and sample:  # get sid from sample
        sid = sample['info']['sid']
    if data_type is None:  # get data_type from sample
        if is_table_sample(sample, sid):
            data_type = 'table'
        elif is_text_sample(sample, sid):
            data_type = 'text'
        else:
            logging.warning("Unknown type of sid: {}".format(sid))

    if data_type.lower().startswith('tab'):
        # filenameId_fileId_pageId_tableIndex_rowcol
        file_id = sid.split('_')[1]
    elif data_type.lower().startswith('text'):
        # fileId_output-???
        file_id = sid.split('_')[0].split('-')[0]
    else:
        file_id = '-1'
    return file_id


def get_document_id(sample=None, sid=None, file_id=None, data_type=None):
    """
    获得 document_id (doc_id / doc_key) 和 file_id 是不一样的东西
    :param sample: 传入一个 UTIE-data 的 dict (sample)
    :param sid: 传入 sid (in sample)
    :param file_id: 传入 file_id (in sid)
    :param data_type: 传入当前 sample/sid 所属的数据类型 (in sid)
    :return: doc_id，通过 doc_map 查表获得
    """
    if file_id:  # direct indexing
        doc_id = DOC_MAP['text2doc'].get(file_id) or DOC_MAP['table2doc'].get(file_id)
        return str(doc_id) or '-1'

    if sid is None and sample:  # get sid from sample
        info = sample['info']
        if 'doc_id' in info:
            return str(info['doc_id'])
        sid = info['sid']
    if data_type is None:  # get data_type from sample
        if is_table_sample(sample, sid):
            data_type = 'table'
        elif is_text_sample(sample, sid):
            data_type = 'text'
        else:
            data_type = 'unknown'

    file_id = get_file_id(
        sample=sample, sid=sid, data_type=data_type)
    if data_type.lower().startswith('tab'):
        doc_id = DOC_MAP['table2doc'].get(file_id)
    elif data_type.lower().startswith('text'):
        doc_id = DOC_MAP['text2doc'].get(file_id)
    else:
        doc_id = DOC_MAP['table2doc'].get(
            file_id, DOC_MAP['text2doc'].get(file_id))
    return str(doc_id) or '-1'


def split_texts_and_tables(data_list):
    """
    将混杂的序列标注数据 sample list 按照 sid 分割成 text 和 table 两个 list
    :param data_list:
    :return:
    """
    from torch.utils.data.dataloader import DataLoader

    if isinstance(data_list, DataLoader):
        texts = [dic[1] for dic in data_list.dataset if not is_table_sample(dic[1])]
        tables = [dic[1] for dic in data_list.dataset if is_table_sample(dic[1])]
    else:  # list of dic
        texts = [dic for dic in data_list if not is_table_sample(dic)]
        tables = [dic for dic in data_list if is_table_sample(dic)]
    return texts, tables


def group_texts_by_doc_id(texts, output):
    """
    将 texts 按照 doc_id 分组
    和 classify_texts 不同 (那个是按 file_id)
    :param texts:
    :param output:
    :return:
    """
    output['texts_keys'] = []
    for sample in texts:
        doc_id = sample['info'].get('doc_id')
        if doc_id is None:
            sid = sample['info']['sid']
            doc_id = get_document_id(sid=sid, data_type='text')
        if doc_id not in output:
            output[doc_id] = []
        if doc_id not in output['texts_keys']:
            output['texts_keys'].append(doc_id)
        # file_id: sample_list
        output[doc_id].append(sample)
    output['texts_keys'] = list(set(output['texts_keys']))
    print('texts_keys:', len(output['texts_keys']))


def group_tables_by_doc_id(tables, output):
    """
    将 tables 按照 doc_id 分组
    和 classify_texts 不同 (那个是按 file_id)
    :param tables:
    :param output:
    :return:
    """
    output['tables_keys'] = []
    for sample in tables:
        doc_id = sample['info'].get('doc_id')
        if doc_id is None:
            sid = sample['info']['sid']
            doc_id = get_document_id(sid=sid, data_type='table')
        if doc_id not in output:
            output[doc_id] = []
        if doc_id not in output['tables_keys']:
            output['tables_keys'].append(doc_id)
        # file_id: sample_list
        output[doc_id].append(sample)
    output['tables_keys'] = list(set(output['tables_keys']))
    print('tables_keys:', len(output['tables_keys']))


def union_entities_for_samples(sl_sample_list, consider_confidence=True, postfix_mark=None, inplace=False):
    """
    对于多个模型预测出的结果，将其合并，entities 取并集
    :param sl_sample_list:
    :param consider_confidence:
    :param postfix_mark:
    :param inplace: replace sid in original samples.
    :return: a list of joined samples
    """
    sid_dict = {}
    if not inplace:
        sl_sample_list = copy.deepcopy(sl_sample_list)
    # group samples by sid
    for sample in sl_sample_list:
        sid = sample['info']['sid']
        if postfix_mark:
            sid = sid[:sid.rindex(postfix_mark)]
        sid_dict.setdefault(sid, [])
        sid_dict[sid].append(sample)

    # union entities for samples sharing the same sid
    for sid in sid_dict:
        leader = sid_dict[sid][0]
        words = leader['words']  # samples with the same sid share words
        pos_dict = {}  # unique entities by position tuples.
        for sample in sid_dict[sid]:
            for ent in sample['entities']:
                ent_pos = get_position_from_entity(ent, words)
                ent_span = tuple([ent_pos[0], ent_pos[-1] + 1])
                if ent_span not in pos_dict:
                    pos_dict[ent_span] = ent
                elif consider_confidence and ('prob' in ent):
                    if ent['prob'] > pos_dict[ent_span]['prob']:
                        pos_dict[ent_span] = ent
                else:  # entities with the same span, we only need one of them
                    continue
        leader['info']['sid'] = sid
        leader['entities'] = [pos_dict[span] for span in sorted(pos_dict.keys())]
        # if pos_dict[span]['type'] in TARGET_CLASSES[1:]]
        leader['relations'] = []
        sid_dict[sid] = leader
    return list(sid_dict.values())


def handle_crossover_in_entities(samples, judge_method='prob'):
    # the samples here are come from union-entities()
    _samples = copy.deepcopy(samples)
    for s_idx, sample in enumerate(_samples):
        words = sample['words']
        pos_dict = {get_position_tuple_from_entity(ent, words): ent
                    for ent in sample['entities']}
        # entities are sorted by pos-tuple
        entities = []
        pivot = (-1, -1)
        pd_keys = sorted([k for k in pos_dict.keys() if pos_dict[k]['type']]) # [(1, 2), (1, 3), (1, 5), (2, 3)]
        for t_idx, tup in enumerate(pd_keys):
            if pos_dict[tup]['type'] == "":
                continue  # meaningless now
            lef, rig = tup
            if lef >= pivot[1]:  # no intersection
                if pivot != (-1, -1):
                    entities.append(pos_dict[pivot])
                pivot = (lef, rig)  # reset current candidate
            elif judge_method == 'len' and rig-lef > pivot[1]-pivot[0]:
                pivot = (lef, rig)  # update current candidate
            elif judge_method == 'prob' and pos_dict[tup]['prob'] > pos_dict[pivot]['prob']:
                pivot = (lef, rig)  # update current candidate
        else:
            if pivot != (-1, -1):
                entities.append(pos_dict[pivot])
        sample['entities'] = entities
    return _samples


def match_doc(doc, mode='11'):
    """
    匹配扩展，为所有 entity_string 找到包含其字符串的文本和表格
    和 recognizer.match_doc 相同，放在这里是为了顺一下流程
    :param doc: Document 类，包括成员变量 sl_text, sl_table, sl_text_dict 和 sl_table_dict
    # :param dicts: doc 的 [sl_text_dict, sl_table_dict]，原意义不明
    :param mode: 两位字符串，第一位表示 doc_library 是否包含 text_dict，第二位表示是否包含 table_dict
    :return: entity 字符串 (set), 实体相关的sid列表 (dict)
    """
    if mode[0] != '0':
        doc.doc_library.update(doc.text_library)
    if mode[1] != '0':
        doc.doc_library.update(doc.table_library)
    entity_library = doc.doc_library

    # list, set, dict, dict
    from col_classification.recognizer_utils import library2groups
    entity_groups, entity_library, text_relevant_sids, table_relevant_sids = library2groups(entity_library)

    for group in entity_groups:
        # 找到所有包含 entity_string 的文本和表格
        text_relevant_sids_part, table_relevant_sids_part = doc.fetch_relevant(
            text_dict=doc.text_dict, table_dict=doc.table_dict, entity_library=group)
        text_relevant_sids.update(text_relevant_sids_part)  # dict
        table_relevant_sids.update(table_relevant_sids_part)  # dict
    return entity_library, text_relevant_sids, table_relevant_sids


def generate_sample_for_cc_model(entity_string, text_dict, table_dict,
                                 text_relevant_sids=None, table_relevant_sids=None):
    """
    基于单个 entity string 生成一个一致性模型的训练数据 sample
    :param entity_string: 单个 entity string
    :param text_dict: 用于处理的数据集 doc.text_dict
    :param table_dict: 用于处理的数据集 doc.table_dict
    :param text_relevant_sids: 带有这个 entity 的 text sid 的 list
    :param table_relevant_sids: 带有这个 entity 的 table sid 的 list
    :return: 一致性模型中可用的单个 sample 数据
        {'sid': sid, 'words': 句子, 'position': [start, end], 'type': 空类型待预测}
    """

    output = {
        'entity': entity_string,
        'sentence': [],
    }
    sentences = output['sentence']
    for _sid, _data_dict in zip([text_relevant_sids, table_relevant_sids],
                                [text_dict, table_dict]):
        if not _sid:  # maybe empty or None
            continue
        for sample_sid in _sid:
            sample = _data_dict[sample_sid]  # {sid: sample}
            for info in sample["relevant_info"][entity_string]:
                item = {
                    'sid': sample_sid,
                    'words': sample['words'],
                    'position': info["span"],
                    'type': info['type']  # mostly empty string
                }
                sentences.append(item)
    return output


def generate_pattern_with_entity_library(entity_library):
    entity_list = [re.escape(entity_string) for entity_string in entity_library]
    sorted_list = sorted(entity_list, key=lambda x: len(x), reverse=True)
    return u'|'.join(sorted_list)


def generate_entity_pattern_by_doc_id(sl_prediction, output_library=True,
                                      output_sids=False, confidence_threshold=0.5):
    entity_sids_by_doc_id = {}
    entity_library_by_doc_id = {}
    for idx, sample in tqdm(enumerate(sl_prediction)):
        words = sample['words']
        sid = sample['info']['sid']
        doc_id = get_document_id(sample=sample)
        for entity in sample['entities']:
            position = get_position_from_entity(entity=entity, words=words)
            ent_type = entity['type']
            ent_prob = float(entity.get('prob', 1.))  # label data doesn't have the key prob
            ent_string = ''.join([words[p]['word'] for p in position])

            if ent_prob >= confidence_threshold:
                # append in sids
                entity_sids_by_doc_id.setdefault(doc_id, {})
                entity_sids_by_doc_id[doc_id].setdefault(ent_string, [])
                entity_sids_by_doc_id[doc_id][ent_string].append(sid)
            # add into entity library
            entity_library_by_doc_id.setdefault(doc_id, {})
            entity_library_by_doc_id[doc_id].setdefault(ent_string, {})
            entity_library_by_doc_id[doc_id][ent_string].setdefault(ent_type, 0)
            entity_library_by_doc_id[doc_id][ent_string][ent_type] += 1

    entity_pattern_by_doc_id = {}
    for doc_id, entity_library in entity_library_by_doc_id.items():
        ent_lib = list(entity_library.keys())
        if '' in ent_lib:
            ent_lib.remove('')
        pat = generate_pattern_with_entity_library(entity_library=ent_lib)
        entity_pattern_by_doc_id[doc_id] = pat

    outputs = (entity_pattern_by_doc_id,)
    if output_library:
        outputs += (entity_library_by_doc_id,)
    if output_sids:
        outputs += (entity_sids_by_doc_id,)
    return outputs


def make_null_entities_sample(samples_list):
    # 制造entities为空，其他不变的集合
    return_list = [{
        "info": copy.deepcopy(sample["info"]),
        "words": copy.deepcopy(sample["words"]),
        "relations": [],
        "entities": []
    } for sample in samples_list]
    return return_list


def clean_blanks_in_sample(sample_list):
    """
    remove useless blanks in sentences
    :param sample_list:
    :return:
    """

    def surround_with_english(words, _idx):
        pre_flag = (_idx > 0) and re.match('^[a-zA-Z]+$', words[_idx - 1]['word'])
        # pre_flag = pre_flag or (_idx == 0)
        post_flag = (_idx < n_words - 1) and re.match('^[a-zA-Z]+$', words[_idx + 1]['word'])
        # post_flag = post_flag or (_idx == n_words - 1)
        if pre_flag and post_flag:
            return True
        return False

    ret = []
    for s_idx, sample in tqdm(enumerate(sample_list)):
        dropped_ids, dropped_indexes = [], []
        n_words = sample['words'].__len__()
        for w_idx, word in enumerate(sample['words']):
            if len(word['word']) == 1 and _is_whitespace(word['word']):
                if surround_with_english(sample['words'], w_idx):
                    continue
                dropped_ids.append(word['id'])
                dropped_indexes.append(w_idx)
        sample['words'] = [w for w in sample['words'] if w['id'] not in dropped_ids]
        for ent_idx, ent in enumerate(sample['entities']):
            ent['tokens'] = [tok for tok in ent['tokens'] if tok not in dropped_ids]
        if 'tag_sequence' in sample:
            tag_size = sample['tag_sequence'].__len__()
            sample['tag_sequence'] = [sample['tag_sequence'][t_idx]
                                      for t_idx in range(tag_size) if t_idx not in dropped_indexes]
        ret.append(sample)
    return ret


def generate_truth_dict(sl_data, dump_path=None, remain_detail=False):
    """
    generate a truth_dict, it returns type label for each (sid, position)
    key is str(pos_tuple), maybe tuple is better, but not stable for json.
    :param sl_data:
    :param dump_path:
    :param remain_detail:
    :return:
    """
    sl_data = copy.deepcopy(sl_data)
    truth = {item['info']['sid']: item for item in sl_data}
    for sid, sample in tqdm(truth.items()):
        sample['positions'] = {}
        for ent in sample['entities']:
            pos = get_position_from_entity(ent, sample['words'])
            pos_tuple = (pos[0], pos[-1] + 1)
            sample['positions'][str(pos_tuple)] = ent['type']
        if not remain_detail:
            truth[sid] = {'positions': sample['positions']}
    if dump_path:
        json.dump(truth, open(dump_path, 'w'))
    return truth


def entity_padding_method(method):
    """
    做 evaluate 的时候，假如两边没有对齐，采用何种 padding 方式进行对齐
    :param method:
    :return:
    """

    def null_padding_by_sid(label_samples, predict_samples):
        """
        predict_samples 里没有的 sid，去 label_samples 里找到对应 sid，将 entities 设置为空
        :param label_samples:
        :param predict_samples:
        :return:
        """
        label_dict = {sample['info']['sid']: sample for sample in label_samples}
        predict_dict = {sample['info']['sid']: sample for sample in predict_samples}
        label_samples = [label_dict[sid] for sid in label_dict]
        predict_samples = [predict_dict[sid] if sid in predict_dict
                           else make_null_entities_sample([label_dict[sid]])[0]
                           for sid in label_dict]
        return label_samples, predict_samples

    def label_padding_by_sid(label_samples, predict_samples):
        """
        predict_samples 里没有的 sid，去 label_samples 里找到对应 sid，设置为 label_samples 里的 entities
        :param label_samples:
        :param predict_samples:
        :return:
        """
        label_dict = {sample['info']['sid']: sample for sample in label_samples}
        predict_dict = {sample['info']['sid']: sample for sample in predict_samples}
        label_samples = [label_dict[sid] for sid in label_dict]
        predict_samples = [predict_dict[sid] if sid in predict_dict
                           else [label_dict[sid]]
                           for sid in label_dict]
        return label_samples, predict_samples

    def label_padding_by_entity(label_samples, predict_samples):
        """
        predict_samples 有的 entity，label_samples 里也有的，设置为 label_samples 里一样的
        :param label_samples:
        :param predict_samples:
        :return:
        """
        label_dict = {sample['info']['sid']: sample for sample in label_samples}
        predict_dict = {sample['info']['sid']: sample for sample in predict_samples}
        label_samples = [label_dict[sid] for sid in label_dict]
        predict_samples = [predict_dict[sid] if sid in predict_dict
                           else [label_dict[sid]]
                           for sid in label_dict]
        return label_samples, predict_samples

    flag_label = 'label' in method
    flag_group = method.split('by_')[-1]

    if not flag_label:
        return null_padding_by_sid

    if flag_group == 'sid':
        return label_padding_by_sid

    if flag_group == 'entity':
        return label_padding_by_entity

    raise ValueError("Invalid method name:", str(method))


def transform_to_origin(data=None, keep_none=False, target_types=TARGET_CLASSES):
    """
    将一致性模型的数据格式(输入/输出均可)，转化为序列标注模型的数据格式
    :param data: cc_sample list
    :return: sl_sample list
    """
    samples = []
    sample_dict = {}
    for tuple1 in data:
        for tuple2 in tuple1['sentence']:
            sid = tuple2['sid']
            if sid not in sample_dict:
                sample_dict[sid] = {
                    u'info': {u'sid': sid},
                    u'words': tuple2['words'],
                    u'entities': [],
                    u'relations': [],
                }
                samples.append(sample_dict[sid])
            sample = sample_dict[sid]
            _type = tuple2['type']
            pos = tuple2['position']
            if _type not in target_types[1:]:
                if not keep_none:
                    continue
                _type = ""
            index2id = {i: word["id"]
                        for i, word in enumerate(sample["words"])}
            insert_entity = {
                u'id': '{}'.format(sample["entities"].__len__()),
                u'tokens': [index2id[j] for j in range(pos[0], pos[1])],
                u'type': _type
            }
            sample["entities"].append(insert_entity)
    return samples


def supervise_by_truth_dict(sl_data, truth_dict, reid=False):
    """
    在 truth_dict 中有的将被更新为其中的答案，没有的则保持原状
    :param sl_data:
    :param truth_dict:
    :param reid:
    :return:
    """
    if reid:
        sl_data, old2new = re_id_func(list(sl_data))
    for idx, sample in tqdm(enumerate(sl_data)):
        sid = sample['info']['sid']
        if sid not in truth_dict:
            continue
        words = sample['words']
        for ent_idx, ent in enumerate(sample['entities']):
            pos = get_position_from_entity(entity=ent, words=words)
            pos_tuple = (pos[0], pos[-1] + 1)
            if str(pos_tuple) in truth_dict[sid]['positions']:
                _type = truth_dict[sid]['positions'][str(pos_tuple)]
                ent['type'] = _type
            # else: remain original label
    return sl_data


def drop_sub_mentions_by_truth_dict(cc_data, truth_dict):
    """
    在 truth_dict 中有的将被更新为其中的答案，没有的则保持原状
    :param cc_data:
    :param truth_dict:
    :return:
    """
    for idx, sample in tqdm(enumerate(cc_data)):
        # ent_string = sample['entity']
        drop_indexes = []
        for ent_idx, can in enumerate(sample['sentence']):
            sid = can['sid']
            pos_tuple = tuple(can['position'])
            for pos_str in truth_dict[sid]['positions']:
                if pos_str == str(pos_tuple):
                    continue  # hit
                if eval(pos_str)[0] <= pos_tuple[0] and pos_tuple[1] <= eval(pos_str)[1]:
                    if truth_dict[sid]['positions'][pos_str] != "":
                        drop_indexes.append(ent_idx)
                        break  # contained
        cc_data[idx]['sentence'] = [sp for i, sp in enumerate(cc_data[idx]['sentence'])
                                    if i not in drop_indexes]
    return cc_data


if __name__ == "__main__":
    print(is_text_sample({'info': {'sid': '901_512'}}))
    print(get_document_id(file_id='901'))

