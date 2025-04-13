





"python -u main.py --language_model 1B --model_path 【】 --hidden_size 3072 --sequence_or_token $sequence_or_token --dataset_name Skywork-Reward-Preference-80K --pipe_data_path 【】 【】 【】 --layer 【】 --latent_size 16384 
    --batch_size 64 --max_length 2048 --lr 5e-4 --betas 0.9 0.999 --num_epochs 1 --seed 42 --steps 10 --use_wandb 1 \
    --pipe_project $train_project $eval_project $pipe_project --device $device --k $k \
    --api_base $api_base --api_key $api_key --api_version $api_version --engine $engine"