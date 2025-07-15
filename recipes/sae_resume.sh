cd src

# GPT interpret Setting
api_base=...
api_key=...
engine=gpt-4o

# Wandb Setting
train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE

# Dataset 
dataset_name=corpus
train_data_path=...
eval_data_path=...
apply_data_path=...

# SAE and LM Backbone Model Setting
k=...
layer=...
hidden_size=...
latent_size=...
model_path=.../meta/Llama-3.1-8B-Instruct/

# Training Setting
# 使用sequence-level还是token level的训练
sequence_or_token=sequence
# 如果apply，使用什么threshold
apply_threshold=3

batch_size=64
# sequence：96 （因为在分句之后，99%的句子长度都在96以下）
max_length=96
device=cuda:1
use_wandb=0
# 0000是字符，0/1表示在[train, evaluate, apply, interpret]这4个位置是否执行，例如1010表示[train, apply]
pipe_run=1000
output_path=../SAE_models
resume_froms=...

echo pipe_run:$pipe_run
echo $sequence_or_token, $layer, $k
echo $hidden_size, $latent_size
python -u main.py --model_path $model_path --hidden_size $hidden_size \
    --pipe_run $pipe_run \
    --resume_form $resume_form \
    --sequence_or_token $sequence_or_token \
    --apply_threshold $apply_threshold \
    --dataset_name $dataset_name \
    --output_path $output_path \
    --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
    --batch_size $batch_size --max_length $max_length --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb $use_wandb \
    --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
    --api_base $api_base --api_key $api_key --engine $engine