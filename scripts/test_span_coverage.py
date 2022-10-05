import jsonlines
from tqdm import tqdm
from collections import Counter, defaultdict


def flatten(nested_list, unique=False):
    ret = [elem for sub_list in nested_list for elem in sub_list]
    if unique:
        return list(set(ret))
    return ret


def count_span_coverage(pure_path, threshold=10):
    ner_length_list = []
    with jsonlines.open(pure_path, 'r') as reader:
        for obj in tqdm(reader):
            _nll = [[ent[1]-ent[0]+1 for ent in sent if len(ent)]
                    for sent in obj["ner"]]
            ner_length_list.extend(flatten(_nll))
        ct = Counter(ner_length_list)

    entities_less_than_10 = sum([v for k, v in ct.items() if k <= threshold])
    entities_all = sum(ct.values())
    print(entities_all-entities_less_than_10, entities_all)
    print(entities_less_than_10 / entities_all, 
          1. - entities_less_than_10 / entities_all)
    print(sorted(ct.keys()))


def count_unique_span_coverage(pure_path, threshold=10):
    unique_entities = defaultdict(set)  # length: a set of (text_spans)
    ner_length_list = []
    with jsonlines.open(pure_path, 'r') as reader:
        for obj in tqdm(reader):
            doc_pivot = 0
            for sent, ner in zip(obj['sentences'], obj['ner']):
                # print(sent, ner)
                if len(ner) != 0:    
                    for _l, _r, _t in ner:
                        text = '_'.join(sent[_l-doc_pivot: _r-doc_pivot+1])
                        unique_entities[_r-_l+1].add(text)
                        # print(text, _l-doc_pivot, _r-doc_pivot+1, sent)
                doc_pivot += len(sent)

    entities_less_than_10 = sum([len(v) for k, v in unique_entities.items() if k <= threshold])
    entities_all = sum(map(len, unique_entities.values()))
    # print(list(unique_entities.items())[:10])
    # for k, v in unique_entities.items():
    #     if k > 45:  print(v)
    print(entities_all-entities_less_than_10, entities_all)
    print(entities_less_than_10 / entities_all, 
          1. - entities_less_than_10 / entities_all)
    print(sorted(unique_entities.keys()))


if __name__ == "__main__":
    # Resume 16 / MSRA 16 / OntoNote4 16 / FinDoc 32
    # pure_path = '/home/chendian/PURE/data/onto4/test.json'
    pure_path = '/home/chendian/PURE/data/findoc/train.json'
    count_span_coverage(pure_path, 10)
    count_unique_span_coverage(pure_path, 10)
