export HF_HOME=/data/zhangsy/.cache
cd src

# GPT interpret Setting
api_base=...
api_key=...
api_version=2024-03-01-preview
engine=gpt-4o

# Wandb Setting
train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE


# Dataset 
dataset_name=Preference232k
train_data_path=...
eval_data_path=...
apply_data_path=...

# SAE and LM Backbone Model Setting
k=...
layer=...
hidden_size=...
latent_size=...
model_path=...

# Training Setting
sequence_or_token=sequence
apply_threshold=5

batch_size=8
# sequence：512 （因为平均而言，对话的长度都很大，所以至少需要512）
max_length=512
device=cuda:0
use_wandb=0
# 0000是字符，0/1表示在[train, evaluate, apply, interpret]这4个位置是否执行，例如1010表示[train, apply]
pipe_run=0010
output_path=...
applied_model=...

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
    --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine --SAE_path $applied_model \
    --split_index 0 --split_num 2 > ../log/1.log 2>&1 &

python -u main.py --model_path $model_path --hidden_size $hidden_size \
    --pipe_run $pipe_run \
    --sequence_or_token $sequence_or_token \
    --apply_threshold $apply_threshold \
    --dataset_name $dataset_name \
    --output_path $output_path \
    --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
    --batch_size $batch_size --max_length $max_length --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb $use_wandb \
    --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
    --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine --SAE_path $applied_model \
    --split_index 1 --split_num 2 > ../log/2.log 2>&1 &