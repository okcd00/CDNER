import os
from pure_api import PureApi


cur_args = {
    'task': 'findoc',
    'max_span_length': 40,
    'boundary_token': 'both',
    'boundary_only_mode': 'span',
}
pa = PureApi(args=cur_args)


def drop_unused_keys():
    for task_name in ['findoc', 'msra_origin', 'cluener', 'onto4', 'resume']:
        for phase in ['train', 'dev', 'test']:
            file_path = f'./data/{task_name}/{phase}.with_span_score.json'
            if os.path.exists(file_path):
                pa.load_from_jsonline(file_path)
                for doc_idx, doc in enumerate(pa.js):
                    if 'predicted_ner' in pa.js[doc_idx]:
                        del pa.js[doc_idx]['predicted_ner']
                    if 'predicted_prob' in pa.js[doc_idx]:
                        del pa.js[doc_idx]['predicted_prob']
                pa.save_as_jsonline(pa.js, file_path)


def generate_score_data():
    for task_name in ['findoc', 'msra_origin', 'cluener', 'onto4', 'resume']:
        for boundary_only_mode in ['bin', 'bdy', 'span']:
            cur_args = {
                'task': task_name,
                'max_span_length': 25,
                'boundary_token': 'both',
                'boundary_only_mode': boundary_only_mode,
            }

            model_dir = '/data/chendian/pure_output_dir'
            load_model_dir = f'{model_dir}/boundary_detection_{task_name}_{boundary_only_mode}/'
            pa = PureApi(args=cur_args, 
                        load_model_dir=load_model_dir)

            for phase in ['train', 'dev', 'test']:
                pa.add_scores_for_pure_samples(
                    file_path=f'./data/{task_name}/{phase}.json',
                    dump_path=f'./data/{task_name}/{phase}.with_{boundary_only_mode}_score.json'
                )


def generate_samples_with_scores():
    import time
    for task_name in ['resume', 'cluener', 'onto4', 'msra_origin', 'findoc']:
        print(task_name)
        for boundary_only_mode in ['span', 'bin', 'bdy']:
            cur_args = {
                'task': task_name,
                'max_span_length': 40 if task_name == 'findoc' else 25,
                'boundary_token': 'both',
                'boundary_only_mode': boundary_only_mode,
            }

            model_dir = '/data/chendian/pure_output_dir'
            load_model_dir = f'{model_dir}/boundary_detection_{task_name}_{boundary_only_mode}_n/'
            pa = PureApi(args=cur_args, 
                        load_model_dir=load_model_dir)

            for phase in ['test', 'train', 'dev']: 
                st = time.time()
                pa.add_scores_for_pure_samples(
                    file_path=f'./data/{task_name}/{phase}.json',
                    dump_path=f'./data/{task_name}/{phase}.with_{boundary_only_mode}_score_n.json'  # _n
                )
                print("cost {:.2f} seconds.".format(time.time()-st))
            print("")