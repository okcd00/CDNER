task_name=msra
score_type=span
score_threshold=250
fusion_method=none

export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=4 python run_entity.py \
    --do_train \
    --do_eval \
    --eval_test \
    --inv_test \
    --take_width_feature True \
    --take_name_module True \
    --take_context_module True \
    --take_context_attn False \
    --take_alpha_loss False \
    --augment_samples True \
    --fusion_method ${fusion_method} \
    --boundary_token both \
    --boundary_only_mode ${score_type}_score \
    --boundary_det_threshold ${score_threshold} \
    --boundary_det_filter_method max \
    --span_filter_method counts \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=4 \
    --eval_batch_size=4 \
    --max_span_length=25 \
    --num_epoch 10 \
    --print_loss_step 500 \
    --model bert-base-chinese \
    --task ${task_name} \
    --data_dir ./data/${task_name} \
    --output_dir /data/chendian/nc_models/${task_name}_${score_type}c_${score_threshold}_wc_${fusion_method}_cover_aug \
