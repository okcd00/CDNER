from tqdm import tqdm
from collections import defaultdict


# in DOC_NER
import sys
import json
import random
import jsonlines
sys.path.append('../doc_ner')
# from data_utils import *
# from file_register import get_path
# from modules.Database import Database
# from scripts import add_span_in_sample


PURE_JSON_HELP = """
A pure-json is like this:
{
  # document ID (please make sure doc_key can be used to identify a certain document)
  "doc_key": "CNN_ENG_20030306_083604.6",

  # sentences in the document, each sentence is a list of tokens
  "sentences": [
    [...],
    [...],
    ["tens", "of", "thousands", "of", "college", ...],
    ...
  ],

  # entities (boundaries and entity type) in each sentence
  "ner": [
    [...],
    [...],
    [[26, 26, "LOC"], [14, 14, "PER"], ...], #the boundary positions are indexed in the document level
    ...,
  ],

  # relations (two spans and relation type) in each sentence
  "relations": [
    [...],
    [...],
    [[14, 14, 10, 10, "ORG-AFF"], [14, 14, 12, 13, "ORG-AFF"], ...],
    ...
  ]
}
"""


def sldb_to_ner_file(sl_db_path, ner_path, delimeter=' '):
    sl_db = Database(sl_db_path)

    with open(ner_path, 'wb') as f:
        for sample in tqdm(sl_db):
            words = [word for word in sample['words']
                     if word['word'].strip()]

            type_tag = ent_type = sample['target_type']
            l, r = sample['target_span']
            span_length = r - l

            for i, word in enumerate(words):
                if i not in range(l, r):
                    t = 'O'
                elif i == l:
                    t = 'B-{}'.format(type_tag)
                    if span_length == 1:
                        t = 'S-{}'.format(type_tag)
                elif i == r - 1:
                    t = 'E-{}'.format(type_tag)
                else:
                    t = 'M-{}'.format(type_tag)
                f.write('{}{}{}\n'.format(
                    word['word'],
                    delimeter,
                    t
                ).encode('utf-8'))
            f.write('\n'.encode('utf-8'))


def ner_to_pure_json(ner_path, pure_path="", tag_dict=None,
                     doc_key="default-doc", max_sentence_length=500, n_samples=-1):
    # ner file for one single pure json

    def _defaultdict_int():
        return defaultdict(int)

    ent_len = defaultdict(_defaultdict_int)
    ent_type_set = set()
    ret = [{"doc_key": doc_key,
            "sentences": [],
            "ner": [],
            "relations": []}]

    def _head(text, c):
        if text.strip().__len__() == 0:
            return False
        return text.lower()[0] in c.lower()

    def _tag_type(tag):
        tp = tag.strip().split('-')[1].strip()
        if tag_dict:
            tp = tag_dict.get(tp, tp)
        return tp

    def generate_sample(char_list, tag_list, offset=0):
        words = char_list
        entities = []

        flag, head = 0, -1
        ent_type = ""
        for idx, tag in enumerate(tag_list):
            if flag == 1:
                if _head(tag, 'e'):
                    # no idx+1 here for PURE's span design
                    entities.append(
                        (head + offset, idx + offset, ent_type))
                    ent_len[ent_type][idx - head + 1] += 1
                    ent_type_set.add(ent_type)
                    flag, head = 0, -1
                    ent_type = ""
                elif _head(tag, 'obs'):
                    entities.append(  # o means ends the last
                        # not idx+1 here for PURE's span design
                        (head + offset, idx + offset - 1, ent_type))
                    ent_len[ent_type][idx - head] += 1
                    ent_type_set.add(ent_type)
                    flag, head = 0, -1
                    ent_type = ""
            if flag == 0:
                if _head(tag, 'b'):
                    flag = 1
                    head = idx
                    ent_type = _tag_type(tag)
                elif _head(tag, 's'):
                    # flag = 0
                    head = idx
                    ent_type = _tag_type(tag)
                    entities.append(
                        (head + offset, idx + offset, ent_type))
                    ent_len[ent_type][1] += 1
                    ent_type_set.add(ent_type)
        else:
            if flag == 1:
                entities.append(
                    (head + offset, idx + offset, ent_type))
                ent_len[ent_type][idx - head + 1] += 1
                ent_type_set.add(ent_type)

        return words, entities

    def add_sample(char_list, tag_list, off):
        if len(ret[0]['sentences']) >= n_samples > 0:
            # no more samples
            return None

        # split too-long sentences
        while len(char_list) > max_sentence_length:
            pivot = max_sentence_length
            while tag_list[pivot - 1] != 'O':
                pivot -= 1
            print("cut at:", char_list[pivot - 2:pivot + 3],
                  tag_list[pivot - 2:pivot + 3])
            off = add_sample(char_list[:pivot], tag_list[:pivot], off)
            char_list, tag_list = char_list[pivot:], tag_list[pivot:]

        words, entities = generate_sample(
            char_list, tag_list, offset=off)
        ret[0]['sentences'].append(words)
        ret[0]['ner'].append(entities)
        ret[0]['relations'].append([])
        off += len(words)
        return off

    offset = 0
    char_list, tag_list = [], []
    for line in tqdm(open(ner_path, 'rb')):
        if line.strip():
            line = line.strip().decode('utf-8').split()
            if len(line) == 1:
                continue
            c, t = line
            char_list.append(c)
            tag_list.append(t)
        else:
            offset = add_sample(char_list, tag_list, offset)
            if offset is None: 
                break  # no more samples, early break
            char_list, tag_list = [], []
    else:
        if len(char_list) + len(tag_list) > 0:
            offset = add_sample(char_list, tag_list, offset)
            char_list, tag_list = [], []

    print("PURE_PATH", pure_path)
    print("")
    for type_name in ent_len:
        print(type_name, ':', sorted(ent_len[type_name].items()))
    print("")
    print("# sentences:", len(ret[0]['ner']))
    print("# words in longest sentence:", max(map(len, ret[0]['sentences'])))
    print(sorted(ent_type_set))
    if pure_path:
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)
    return ret


def sl_db_to_pure_json():
    dataset_name = 'data'
    for phase in ['train', 'valid', 'test']:
        sl_db_path = get_path('sl_{}_{}'.format(dataset_name, phase))
        # sl_db_path = DATA_PATH + 'sl_{}5texts_{}_folder/'.format(phase, dataset_name)
        print("SLDB_PATH", sl_db_path)

        ret = []
        db = Database(sl_db_path)
        doc2sid = {}
        sid2idx = {}
        for idx, sample in enumerate(db):
            doc = sample['info'].get('doc_id', 'doc-0')
            sid = sample['info']['sid']
            doc2sid.setdefault(doc, [])
            doc2sid[doc].append(sid)
            sid2idx[sid] = idx

        for doc, sid_list in tqdm(doc2sid.items()):
            doc_pivot = 0
            sample = {"doc_key": doc, "sentences": [], "ner": [], "relations": []}
            for sid in sid_list:
                _sp = add_span_in_sample(db[sid2idx[sid]])
                sample["sentences"].append([w['word'] for w in _sp['words']])
                ner_list = []
                for ent in _sp['entities']:
                    l, r = ent['span']
                    r -= 1  # in DQ Chen's PURE project, span is [l, r0]
                    ent_type = ent['type']
                    ner_list.append([l+doc_pivot, r+doc_pivot, ent_type])
                sample["ner"].append(ner_list)
                sample["relations"].append([])
                doc_pivot += len(_sp['words'])
            ret.append(sample)

        pure_path = '/home/chendian/PURE/data/{}/{}.json'.format(
            dataset_name.replace('data', 'findoc'), phase.replace('valid', 'dev'))
        print("PURE_PATH", pure_path)
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)


def mrc_ner_to_pure_json(dir_path='/home/chendian/download', dataset_name='ace04'):
    tag_dict = ['FAC', 'WEA', 'LOC', 'VEH', 'GPE', 'ORG', 'PER']
    tag_case = set()
    for phase in ['train', 'dev', 'test']:
        json_path = f"{dir_path}/{dataset_name}/mrc-ner.{phase}"
        print("MRC_File_PATH", json_path)

        ret = []
        reader = json.load(open(json_path, "r"))

        doc_pivot = 0
        sample = {"doc_key": 0, "sentences": [], "ner": [], "relations": []}
        
        last_qas_id = "?.?"
        words = []
        ner_list = []
        for item in tqdm(reader):
            qas_id = item['qas_id']
            if qas_id.split('.')[0] != last_qas_id.split('.')[0]:
                if last_qas_id != "?.?":
                    sample["sentences"].append(words)
                    sample['ner'].append(ner_list)
                    sample['relations'].append([])
                last_qas_id = f"{qas_id}"
                doc_pivot += len(words)  # new offset
                words = [w for w in item['context'].split(' ') if w]  # new words
                ner_list = []  # clear ner_list
            
            tag_name = item['entity_label']
            tag_case.add(tag_name)
            # tag_name = tag_dict.get(tag)
            for pos_str in item['span_position']:
                lef, rig = pos_str.split(';')
                lef, rig = int(lef), int(rig)
                ner_list.append([lef+doc_pivot, rig+doc_pivot, tag_name])
        else:
            sample["sentences"].append(words)
            sample['ner'].append(ner_list)
            sample['relations'].append([])
            
        ret.append(sample)
        pure_path = f'/home/chendian/PURE/data/{dataset_name}/{phase}.json'
        print("PURE_PATH", pure_path)
        print(tag_case)
        print(f"{sum(map(len, ret[0]['ner']))} entities in {len(ret[0]['sentences'])} sentences")
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)


def cluener_to_pure_json(dir_path):
    tag_dict = {
        'address': '地址',
        'book': '书名',
        'company': '公司',
        'game': '游戏',
        'government': '政府',
        'movie': '电影',
        'name': '姓名',
        'organization': '机构',
        'position': '职位',
        'scene': '景点'
    }

    """
    地址（address）: **省**市**区**街**号，**路，**街道，**村等（如单独出现也标记），注意：地址需要标记完全, 标记到最细。
    书名（book）: 小说，杂志，习题集，教科书，教辅，地图册，食谱，书店里能买到的一类书籍，包含电子书。
    公司（company）: **公司，**集团，**银行（央行，中国人民银行除外，二者属于政府机构）, 如：新东方，包含新华网/中国军网等。
    游戏（game）: 常见的游戏，注意有一些从小说，电视剧改编的游戏，要分析具体场景到底是不是游戏。
    政府（government）: 包括中央行政机关和地方行政机关两级。 中央行政机关有国务院、国务院组成部门（包括各部、委员会、中国人民银行和审计署）、国务院直属机构（如海关、税务、工商、环保总局等），军队等。
    电影（movie）: 电影，也包括拍的一些在电影院上映的纪录片，如果是根据书名改编成电影，要根据场景上下文着重区分下是电影名字还是书名。
    姓名（name）: 一般指人名，也包括小说里面的人物，宋江，武松，郭靖，小说里面的人物绰号：及时雨，花和尚，著名人物的别称，通过这个别称能对应到某个具体人物。
    组织机构（organization）: 篮球队，足球队，乐团，社团等，另外包含小说里面的帮派如：少林寺，丐帮，铁掌帮，武当，峨眉等。
    职位（position）: 古时候的职称：巡抚，知州，国师等。现代的总经理，记者，总裁，艺术家，收藏家等。
    景点（scene）: 常见旅游景点如：长沙公园，深圳动物园，海洋馆，植物园，黄河，长江等。
    """

    dataset_name = 'cluener'
    tag_case = set()
    for phase in ['train', 'dev', 'test']:
        json_path = f"{dir_path}/{phase}.jsonl"
        print("ClueNER_File_PATH", json_path)

        ret = []
        with jsonlines.open(json_path, "r") as reader:
            doc_pivot = 0
            sample = {"doc_key": 0, "sentences": [], "ner": [], "relations": []}
            for item in tqdm(reader):
                ner_list = []
                sample["sentences"].append([w for w in item['text']])
                # "name": {"Riddick": [[9, 15]], "Johns": [[28, 32]]}
                for tag, entities in item.get('label', {}).items():
                    tag_name = tag_dict.get(tag)
                    tag_case.add(tag_name)
                    for span_text, span_position in entities.items():
                        lef, rig = span_position[0]  # [[6, 8]]
                        ner_list.append([lef+doc_pivot, rig+doc_pivot, tag_name])
                sample['ner'].append(ner_list)
                sample['relations'].append([])
                doc_pivot += len(item['text'])
            ret.append(sample)

        pure_path = f'/home/chendian/PURE/data/cluener/{phase}.json'
        print("PURE_PATH", pure_path)
        print(tag_case)
        print(f"{sum(map(len, ret[0]['ner']))} entities in {len(ret[0]['sentences'])} sentences")
        with jsonlines.open(pure_path, mode='w') as writer:
            writer.write_all(ret)


def weibo_to_ner():
    """
    人名 : [(1, 9), (2, 178), (3, 73), (4, 16), (5, 2), (7, 1), (8, 1), (11, 1)]
    位置 : [(2, 38), (3, 8), (4, 2), (5, 1)]
    公司 : [(2, 24), (3, 18), (4, 7), (5, 1), (6, 4), (8, 1), (11, 1)]
    地址 : [(2, 10), (3, 8), (4, 7), (5, 2), (6, 1)]
    """
    labels = set()
    mapping = {
        'GPE.NAM': '位置',
        'GPE.NOM': '位置',
        'LOC.NAM': '地址',
        'LOC.NOM': '地址',
        'ORG.NAM': '公司',
        'ORG.NOM': '公司',
        'PER.NAM': '人名',
        'PER.NOM': '人名',
    }
    for phase in ['train', 'dev', 'test']:
        with open(f'/home/chendian/PURE/data/weibo/{phase}.ner', 'w') as f:
            for line in open(f'/data/chendian/download/weibo/weiboNER_2nd_conll.{phase}', 'r'):
                if not line.strip():
                    f.write("\n")
                else:
                    items = line.strip().split()
                    tok, label = items[0][0], items[1]
                    if '-' in label:
                        label = label[:2] + mapping.get(label[2:])
                    labels.add(label)
                    _str = f'{tok}\t{label}'
                    f.write(f"{_str}\n")
    print(sorted(labels))


def onto5_to_ner():
    labels = set()
    has_blank_line = True
    for phase in ['train', 'dev', 'test']:
        with open(f'/home/chendian/PURE/data/onto5/{phase}.ner', 'w') as f:
            for line in open(f'/data/chendian/download/onto5/result/chinese/ontonotes5.{phase}.bmes', 'r'):
                if not line.strip():
                    if not has_blank_line:
                        f.write("\n")
                        has_blank_line = True
                else:
                    items = line.strip().split()
                    tok, label = items[0][0], items[1]
                    # if '-' in label:  label = label[:2] + label[2:]
                    labels.add(label)
                    _str = f'{tok}\t{label}'
                    f.write(f"{_str}\n")
                    has_blank_line = False
    print(sorted(labels))


def test_for_common_dataset():
    msra_origin_tag_dict = {'PER': '人名', 'LOC': '地址', 'ORG': '公司'}
    onto4_tag_dict = {'PER': '人名', 'LOC': '地址', 'ORG': '公司', 'GPE': '位置'}
    resume_tag_dict = {
        'ORG': '公司',
        'EDU': '学历',
        'PRO': '专业',
        'LOC': '地址',
        'NAME': '人名',
        'CONT': '国籍',
        'RACE': '民族',
        'TITLE': '职称',
    }

    phase = 'train'
    dataset_name = 'msra'
    ret_test = ner_to_pure_json(
        ner_path='/home/chendian/PURE/data/{}_origin/{}.ner'.format(dataset_name, phase),
        pure_path='/home/chendian/PURE/data/{}_small/{}.json'.format(dataset_name, phase),
        # tag_dict={'NR': '人名', 'NS': '地址', 'NT': '公司'},  # msra
        tag_dict=msra_origin_tag_dict,  # msra-origin
        # tag_dict=resume_tag_dict,
        # tag_dict=onto4_tag_dict,  # onto4
        doc_key=dataset_name + '-doc',
        # n_samples=10000 if phase == 'train' else -1
    )


def sample_small_trainset(pure_path, dump_path=None, n_samples=10000):
    ret = []
    with jsonlines.open(pure_path, mode='r') as reader:
        for doc in reader:
            # offset = 0
            offset_delta = 0
            indexes = random.sample(range(len(doc['sentences'])), n_samples)
            new_ner_case = []
            for i, ner_items in enumerate(doc['ner']): 
                if i in indexes:
                    # offset += len(doc['sentences'][i])
                    new_ner_case.append([ [lef-offset_delta, rig-offset_delta, tag] for lef, rig, tag in ner_items ])
                else:
                    offset_delta += len(doc['sentences'][i])
            doc['ner'] = new_ner_case
            doc['sentences'] = [sample for i, sample in enumerate(doc['sentences']) if i in indexes]
            doc['relations'] = [[] for _ in range(n_samples)]
            ret.append(doc)
    print(len(ret[0]['sentences']))

    if dump_path is None:
        dump_path = pure_path.replace('.json', '_small.json')
    with jsonlines.open(dump_path, mode='w') as writer:
        writer.write_all(ret)


def generate_boundary_only_dataset(pure_path, dump_path=None):
    ret = []
    with jsonlines.open(pure_path, mode='r') as reader:
        for doc in reader:
            for sent_idx, sent in enumerate(doc['ner']):
                doc['ner'][sent_idx] = [(_l, _r, '实体') for _l, _r, _t in sent]
            ret.append(doc)

    if dump_path is None:
        dump_path = pure_path.replace('.json', '.bdy_only.json')
    with jsonlines.open(dump_path, mode='w') as writer:
        writer.write_all(ret)


if __name__ == "__main__":
    # cluener_to_pure_json(dir_path='/home/chendian/PURE/data/cluener/')
    # mrc_ner_to_pure_json(dataset_name='conll03')
    # sample_small_trainset(
    #     '/home/chendian/PURE/data/msra_origin/train.json', 
    #     '/home/chendian/PURE/data/msra_xs/train.json', 5000)
    # generate_boundary_only_dataset('/home/chendian/PURE/data/msra_origin/train.json')
    # generate_boundary_only_dataset('/home/chendian/PURE/data/msra_origin/dev.json')
    onto5_to_ner()
    for phase in ['train', 'dev', 'test']:
        ner_to_pure_json(f'/home/chendian/PURE/data/onto5/{phase}.ner', 
                         pure_path=f'/home/chendian/PURE/data/onto5/{phase}.json', 
                         tag_dict=None,
                         doc_key="onto5-doc", max_sentence_length=500, n_samples=-1)
    
