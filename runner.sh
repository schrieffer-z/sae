cd src
export WANDB_API_KEY='ac8217b0848b0b74ed1f9abd8bee6b09afcc7b5c'
api_base=https://dplc-8.openai.azure.com/
api_key=DcuVRqY7eBsgOHcO5IvvRShkDKlWhJEDm2ZRHHU2Ja3O8c3HBvT4JQQJ99AKACHrzpqXJ3w3AAABACOGvoip
api_version=2024-03-01-preview
engine=gpt-4o

model_path_prefix=/mnt/finder/lisihang/models/meta/
train_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/10M
eval_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata
apply_data_path=/mnt/finder/lisihang/xAI-RLHF/Shuyi/sae/data/testdata

train_project=SAE4RM-train-SAE
eval_project=SAE4RM-eval-SAE
pipe_project=SAE4RM-pipe-SAE

device=cuda:7
        
for sequence_or_token in "sequence"; do
    for layer in 8 12 16; do
        echo $sequence_or_token,$layer

        python -u main.py --language_model 1B --model_path $model_path_prefix"Llama-3.2-1B-Instruct" --hidden_size 2048 \
            --sequence_or_token $sequence_or_token\
            --pipe_data_path $train_data_path $eval_data_path $apply_data_path --layer $layer --latent_size 16384 \
            --batch_size 64 --max_length 96 --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
            --pipe_project $train_project $eval_project $pipe_project --device $device --k 100 \
            --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine 
    done
done