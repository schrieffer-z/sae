cd src

# GPT interpret Setting
api_base==...
api_key=...
api_version=2024-03-01-preview
engine=gpt-4o

# Wandb Setting
train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE
export WANDB_API_KEY='ac8217b0848b0b74ed1f9abd8bee6b09afcc7b5c'

# Dataset 
dataset_name=OpenWebText
train_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/50M
eval_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata
apply_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata

# SAE and LM Backbone Model Setting
k=...
layer=...
hidden_size=...
latent_size=...
model_path=/mnt/finder/lisihang/models/google/gemma-2-2b-it/

# Training Setting
batch_size=64
max_length=96
device=cuda:1
use_wandb=0
pipe_run=1000
output_path=../SAE_models

echo pipe_run:$pipe_run
echo $sequence_or_token, $layer, $k
echo $hidden_size, $latent_size
python -u main.py --model_path $model_path --hidden_size $hidden_size \
    --pipe_run $pipe_run \
    --sequence_or_token token \
    --dataset_name $dataset_name \
    --output_path $output_path \
    --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
    --batch_size $batch_size --max_length $max_length --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb $use_wandb \
    --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
    --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine

