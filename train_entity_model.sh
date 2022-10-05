task_name=msra
# --model /data/chendian/pretrained_bert_models/chinese_L-12_H-768_A-12/ \
# --model /data/chendian/pretrained_bert_models/chinese-macbert-base/ \

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python run_entity.py \
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
    --train_batch_size=16 \
    --eval_batch_size=16 \
    --max_span_length=25 \
    --num_epoch=20 \
    --print_loss_step=500 \
    --context_window=0 \
    --filtering_strategy ratio-20 \
    --model /data/chendian/pretrained_bert_models/chinese-macbert-base/ \
    --task ${task_name} \
    --data_dir ./data/${task_name} \
    --output_dir /data/chendian/pure_output_dir/${task_name}_macB_ratio20_221006 \
