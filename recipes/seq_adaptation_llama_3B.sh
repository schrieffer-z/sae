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
export WANDB_API_KEY=...

# Dataset 
dataset_name=Skywork-Reward-Preference-80K
train_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/Skywork-train
eval_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/Skywork-eval
apply_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/Skywork-eval

# LM backbone for hidden state
model=meta/Llama-3.2-3B-Instruct
model_path_prefix=/mnt/finder/lisihang/models/

# SAE Training Setting
sequence_or_token=sequence
batch_size=64
max_length=2048
device=cuda:1
resume_froms+=(../SAE_models/token_Latent16384_Layer8_K32_1B_10M)
resume_froms+=(../SAE_models/token_Latent16384_Layer12_K64_1B_10M)
resume_froms+=(../SAE_models/token_Latent16384_Layer8_K32_1B_10M)
resume_froms+=(../SAE_models/token_Latent16384_Layer12_K64_1B_10M)

for resume_from in ${resume_froms[@]}; do
    echo $sequence_or_token, $layer, $k
    echo $model_size, $hidden_size, $latent_size
    python -u main.py --model_size $model_size --model_path $model_path_prefix$model --hidden_size $hidden_size \
        --sequence_or_token $sequence_or_token \
        --dataset_name $dataset_name \
        --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
        --batch_size $batch_size --max_length $max_length --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
        --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
        --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine \
        --resume_from $resume_from.pt 
done
