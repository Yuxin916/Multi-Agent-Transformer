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
from mat.envs.ma_mujoco.multiagent_mujoco.mujoco_multi import MujocoMulti
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space

def parse_args(args, parser):
    parser.add_argument('--scenario', type=str, default='Hopper-v2', help="Which mujoco task to run on")
    parser.add_argument('--agent_conf', type=str, default='3x1')
    parser.add_argument('--agent_obsk', type=int, default=0)
    parser.add_argument("--faulty_node",  type=int, default=-1)
    parser.add_argument("--eval_faulty_node", type=int, nargs='+', default=None)
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)

    # agent-specific state should be designed carefully
    parser.add_argument("--use_state_agent", action='store_true', default=False)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=False)

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True

    # env
    env_args = {"scenario": all_args.scenario,
                "agent_conf": all_args.agent_conf,
                "agent_obsk": all_args.agent_obsk,
                "episode_limit": 1000}

    env = MujocoMulti(env_args=env_args)
    obs_dim = get_shape_from_obs_space(env.observation_space[0])[0]
    share_obs_dim = get_shape_from_obs_space(env.share_observation_space[0])[0]
    act_dim = env.action_space[0].shape[0]
    num_agents = env.n_agents

    print("num_agents: ", num_agents)
    print("obs_dim: ", obs_dim)
    print("share_obs_dim: ", share_obs_dim)
    print("act_dim: ", act_dim)
    print("continuous_action_space")

    pass




if __name__ == "__main__":
    main(sys.argv[1:])