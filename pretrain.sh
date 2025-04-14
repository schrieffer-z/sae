cd src
export WANDB_API_KEY='ac8217b0848b0b74ed1f9abd8bee6b09afcc7b5c'
api_base=https://dplc-8.openai.azure.com/
api_key=DcuVRqY7eBsgOHcO5IvvRShkDKlWhJEDm2ZRHHU2Ja3O8c3HBvT4JQQJ99AKACHrzpqXJ3w3AAABACOGvoip
api_version=2024-03-01-preview
engine=gpt-4o

train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE

dataset_name=OpenWebText
train_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/10M
eval_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata
apply_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata

model_path_prefix=/mnt/finder/lisihang/models/
model=meta/Llama-3.2-1B-Instruct
batch_size=128
max_length=96
device=cuda:1

for sequence_or_token in "token"; do
    for layer in 8 12; do
        for k in 96 128 256; do
            if echo "$model" | grep -q "1B";then
                model_size=1B
                hidden_size=2048
                latent_size=16384
            elif echo "$model" | grep -q "3B";then
                model_size=3B
                hidden_size=3072
                latent_size=24576
            fi

            echo $sequence_or_token, $layer, $k
            echo $model_size, $hidden_size, $latent_size
            python -u main.py --model_size $model_size --model_path $model_path_prefix$model --hidden_size $hidden_size \
                --sequence_or_token $sequence_or_token \
                --dataset_name $dataset_name \
                --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size $latent_size \
                --batch_size $batch_size --max_length $batch_size --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
                --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
                --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine \
        done
    done
done
