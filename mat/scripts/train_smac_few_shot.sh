#!/bin/bash
"""
It is employed to fine-tune the pre-trained model for every single task one by one
"""
env="StarCraft2_multi"
maps=("27m_vs_30m" "8m" "5m_vs_6m" "MMM2" "1c3s5z" "2s_vs_1sc")
algo="mat"
exp="from_scratch_"
seed=1
n_rollout_threads=8
n_eval_rollout_threads=8

# 这里和train_smac_multi.sh不同，for遍历一个一个环境，所以只需要一个线程
for map in "${maps[@]}"
do
  exp_map=$exp$map
  echo "env is ${env}, train_maps is ${map}, algo is ${algo}, exp is ${exp_map}, seed is ${seed}"
  CUDA_VISIBLE_DEVICES=0 python train/train_smac_multi.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp_map} --train_maps ${map} --eval_maps ${map} --seed ${seed} --n_eval_rollout_threads ${n_eval_rollout_threads} --n_rollout_threads ${n_rollout_threads} --episode_length 100 -- 5 --num_env_steps 1000000 --ppo_epoch 10 --clip_param 0.05 --use_eval
done
