#!/bin/sh
"""
It is used to pre-train multiple maps simultaneously, which generates a pre-trained model for different tasks
"""
env="StarCraft2_multi"
#train_maps="3s_vs_3z 3s_vs_4z 3m MMM 3s5z 8m_vs_9m 25m 10m_vs_11m 2s3z"
#eval_maps="3s_vs_3z 3s_vs_4z 3m MMM 3s5z 8m_vs_9m 25m 10m_vs_11m 2s3z"
train_maps="3s_vs_3z 8m_vs_9m 10m_vs_11m 2s3z"
eval_maps="3s_vs_3z 8m_vs_9m 10m_vs_11m 2s3z"
algo="mat"
exp="multi_task"
seed=1
n_rollout_threads=8
n_eval_rollout_threads=8

# change the parallel environment number according to your CPU setting
echo "env is ${env}, train_maps is ${train_maps}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
"""
仔细检查所有的parameter
1. --n_rollout_threads是平行环境的数量，这个数量要和train_maps的数量一致/是他的倍数
2. --n_eval_rollout_threads的数量也要和train_maps的数量一致/是他的倍数
"""
CUDA_VISIBLE_DEVICES=0 python train/train_smac_multi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --train_maps ${train_maps} --eval_maps ${eval_maps} --seed ${seed} --n_eval_rollout_threads ${n_eval_rollout_threads} --n_rollout_threads ${n_rollout_threads} --episode_length 100 --ppo_epoch 10 --clip_param 0.05 --use_eval
