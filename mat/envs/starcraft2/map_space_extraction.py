#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from mat.config import get_config
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from mat.envs.starcraft2.smac_maps import get_map_params
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.envs.starcraft2.smac_maps import get_smac_map_registry


def parse_args(args, parser):
    parser.add_argument('--map_name', type=str, default='3m', help="Which smac map to run on")
    parser.add_argument('--eval_map_name', type=str, default='3m', help="Which smac map to eval on")
    parser.add_argument('--run_dir', type=str, default='', help="Which smac map to eval on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)
    parser.add_argument("--random_agent_order", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # 从config.py中获取参数
    parser = get_config()
    # 从parse_args命令行中获取参数
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mat_dec":
        # 默认在mat_dec中actor之间共享参数
        all_args.dec_actor = True
        all_args.share_actor = True

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 创建训练环境的路径
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    all_args.run_dir = run_dir

    map_param_registry = get_smac_map_registry()
    for ele in map_param_registry.keys():
        all_args.map_name = ele

        env = StarCraft2Env(all_args)
        # print("obs_space: ", env.observation_space)
        # print("share_obs_space: ", env.share_observation_space)
        # print("act_space: ", env.action_space)
        obs_dim = get_shape_from_obs_space(env.observation_space[0])[0]
        share_obs_dim = get_shape_from_obs_space(env.share_observation_space[0])[0]
        act_dim = env.action_space[0].n

        print("map", all_args.map_name)
        print("obs_dim: ", obs_dim)
        print("share_obs_dim: ", share_obs_dim)
        print("act_dim: ", act_dim)

        # write to txt file
        with open("map_space.txt", "a") as f:
            f.write("map: " + all_args.map_name + "\n")
            f.write("obs_dim: " + str(obs_dim) + "\n")
            f.write("share_obs_dim: " + str(share_obs_dim) + "\n")
            f.write("act_dim: " + str(act_dim) + "\n")
            f.write("\n")

    pass

if __name__ == "__main__":
    main(sys.argv[1:])
