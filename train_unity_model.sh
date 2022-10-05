task_name=unity

export PYTHONPATH=.
python run_entity.py \
    --do_train \
    --do_eval \
    --eval_test \
    --learning_rate=1e-5 \
    --task_learning_rate=5e-4 \
    --train_batch_size=8 \
    --eval_batch_size=8 \
    --max_span_length=40 \
    --num_epoch 20 \
    --print_loss_step 500 \
    --context_window 0 \
    --task ${task_name} \
    --data_dir ./data/${task_name}_with_findoc \
    --model bert-base-chinese \
    --bert_model_dir /data/chend/model_files/chinese_L-12_H-768_A-12/ \
    --output_dir ./output_dir/${task_name}_with_findoc
