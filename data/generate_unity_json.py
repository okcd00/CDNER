import json
import jsonlines


for phase in ['train', 'dev', 'test']:
    dict_list = []
    for dataset_name in ['msra', 'resume', 'onto4', 'findoc']:
        jsonl_file = '/home/chendian/PURE/data/{}/{}.json'.format(dataset_name, phase)
        print("Analysis on {}".format(jsonl_file))
        js = [json.loads(line) for line in open(jsonl_file, 'r')]
        # for s, n, r in zip(line['sentences'], line['ner'], line['relations']):
        #     n = [[lef, rig, "实体"] for lef, rig, tp in tp in []]
        for line in js:
            line['ner'] = [
                [[lef, rig, "实体"] for lef, rig, tp in tags
                                    if tp in ['人名', '地址', '公司', '位置']]
                 for tags in line['ner']]
            dict_list.append(line)
    save_path = '/home/chendian/PURE/data/unity_with_findoc/{}.json'.format(phase)
    with jsonlines.open(save_path, mode='w') as writer:
        writer.write_all(dict_list)
