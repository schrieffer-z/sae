cd src

# GPT interpret Setting
api_base=
api_key=
engine=gpt-4o

# Wandb Setting
train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE

# Training Setting
apply_threshold=5




use_wandb=0
pipe_run=0001
output_path=...
selected_latent_path=../llama8b_sequence_Latent65536_Layer16_K192_1B.json

echo pipe_run:$pipe_run
echo $sequence_or_token, $layer, $k
echo $hidden_size, $latent_size
python -u main.py --model_path $model_path --hidden_size $hidden_size \
    --pipe_run $pipe_run \
    --sequence_or_token $sequence_or_token \
    --apply_threshold $apply_threshold \
    --dataset_name $dataset_name \
    --output_path $output_path \
    --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
    --batch_size $batch_size --max_length $max_length --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb $use_wandb \
    --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
    --api_base $api_base --api_key $api_key --engine $engine --selected_latent_path $selected_latent_path