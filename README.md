# CDNER
The released code for:
1) the Candidate Distilled CNER model.
2) the Context Deserved CNER model (NCModel).


# Usage

## Script
```bash
task_name=msra  # [msra/resume/onto4/onto5/weibo]

# path to BERT/MacBERT, change it to yours
# --model /data/chendian/pretrained_bert_models/chinese_L-12_H-768_A-12/ \
# --model /data/chendian/pretrained_bert_models/chinese-macbert-base/ \

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=2 python run_entity.py \
    --do_train \
    --do_eval \
    --eval_test \
    --take_width_feature True \
    --take_name_module True \
    --take_context_module False \
    --take_context_attn False \
    --boundary_token both \
    --boundary_only_mode none \
    --fusion_method none \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --span_filter_method none \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --max_span_length=25 \
    --num_epoch=20 \
    --print_loss_step=500 \
    --context_window=0 \
    --filtering_strategy ratio-20 \
    --model /data/chendian/pretrained_bert_models/chinese_L-12_H-768_A-12/ \
    --task ${task_name} \
    --data_dir ./data/${task_name} \
    --output_dir /data/chendian/pure_output_dir/${task_name}_B_ratio20_221012 \
```

### Start Training

```bash
# modify the shell command at first
sh train_entity_model.sh
```

### Start Evaluating

```bash
# remove --do_train
sh train_entity_model.sh
```

## Results

> Experimental Results for $\theta_{FRNP} = 20$

Datasets  |  MSRA   | Resume | Onto4 | Weibo | Onto5 |
---- |  ----   | ----   | ----  | ----  | ----  |
**Precision** | 96.68 | 96.57 | 82.15 | 75.85 | 75.81 |
**Recall** | 95.53 | 96.75 | 80.99 | 69.81 | 82.17 |
**F1-Score** | 96.10 | 96.66 | 81.57 | 72.70 | 78.86 |