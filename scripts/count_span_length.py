import jsonlines
from tqdm import tqdm
from pprint import pprint
from collections import defaultdict
from entity.utils import load_json, dump_json


def count_span_length(data_dir_name):
    phases = ['train', 'dev', 'test']
    TAG_MAPPING = {"位置": "地址", "姓名": "人名", "机构": "公司"}
    TAG_MAPPING.update({k: k for k in ['人名', '公司', '地址']})

    span_count_dict = {}
    for phase in phases:
        filename = f"./data/{data_dir_name}/{phase}.json"

        counts = defaultdict(dict)  # counts[span_length][ent_type] = value
        with jsonlines.open(filename, 'r') as reader:
            for doc in tqdm(reader):
                for sent in doc["ner"]:
                    for lef, rig, tag in sent:
                        _tag = TAG_MAPPING.get(tag, '其它')
                        # lef == rig for single-token word (lef, rig) in PURE
                        counts[rig-lef+1].setdefault(_tag, 0)
                        counts[rig-lef+1][_tag] += 1
        # pprint(counts)
        dumpfile = f"./data/{data_dir_name}/{phase}.span_counts.json"
        print(f"Now dump count_info into {dumpfile}")
        dump_json(counts, dumpfile)
        span_count_dict[phase] = counts
    
    dumpfile = f"./data/{data_dir_name}/total.span_counts.json"
    print(f"Now dump count_info into {dumpfile}")
    dump_json(span_count_dict, dumpfile)


if __name__ == "__main__":
    target_dir_names = ['msra_origin', 'onto4', 'resume', 'cluener', 'findoc']
    # target_dir_names = ['conll03', 'ace04', 'ace05', 'scierc']
    for dir_name in target_dir_names:
        count_span_length(dir_name)
    
