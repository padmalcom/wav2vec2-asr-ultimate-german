export LR=5e-5
export ACC=4 # batch size * acc = 8
export WORKER_NUM=1

bitfusion run -n 1 "python train.py \
--output_dir=out5/ \
--num_train_epochs=200 \
--per_device_train_batch_size='2' \
--per_device_eval_batch_size='2' \
--gradient_accumulation_steps=$ACC \
--evaluation_strategy='steps' \
--save_total_limit='1' \
--save_steps='1000' \
--eval_steps='5000' \
--logging_steps='500' \
--logging_dir='log' \
--do_train \
--do_eval \
--learning_rate=$LR \
--dataloader_num_workers $WORKER_NUM"