export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python run_multiple_choice.py \
  --do_lower_case \
  --do_train \
  --do_eval \
  --do_test \
  --overwrite_output \
  --eval_all_checkpoints \
  --task_name dream \
  --per_gpu_eval_batch_size=24 \
  --logging_steps 20 \
  --max_seq_length 512 \
  --model_type albert \
  --model_name_or_path albert-xxlarge-v2 \
  --data_dir ../dream_data \
  --learning_rate 1e-5 \
  --num_train_epochs 4 \
  --output_dir albert_xxlarge_dream_lr1e5_ep4 \
  --per_gpu_train_batch_size=1 \
  --gradient_accumulation_steps 1 \
  --warmup_steps 300 \
  --save_steps 400