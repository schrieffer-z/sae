cd src

# GPT interpret Setting
api_base==...
api_key=...
api_version=...
engine=...

# Wandb Setting
train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE
export WANDB_API_KEY='ac8217b0848b0b74ed1f9abd8bee6b09afcc7b5c'

# Dataset 
dataset_name=OpenWebText
train_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/100M
eval_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata
apply_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata

# LM backbone for hidden state
hidden_size=2048
latent_size=16384
model=meta/Llama-3.2-1B-Instruct
model_path_prefix=/mnt/finder/lisihang/models/

# SAE Training Setting
batch_size=64
max_length=96
device=cuda:0

for sequence_or_token in "token"; do
    for layer in 8 10 12; do
        for k in 32 64 96 128; do
            echo $sequence_or_token, $layer, $k
            echo $model_size, $hidden_size, $latent_size
            python -u main.py --model_path $model_path_prefix$model --hidden_size $hidden_size \
                --sequence_or_token $sequence_or_token \
                --dataset_name $dataset_name \
                --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
                --batch_size $batch_size --max_length $batch_size --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
                --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
                --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine
        done
    done
done
