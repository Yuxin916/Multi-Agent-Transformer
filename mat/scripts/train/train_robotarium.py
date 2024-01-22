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
import yaml
from functools import partial
from multiprocessing import Pipe, Process
from types import SimpleNamespace as SN
from mat.runner.shared.robotarium_runner import RobotariumRunner as Runner
from mat.algorithms.mat.env_tools import make_train_env_robo, make_eval_env_robo
"""Train script for Robotarium."""


def parse_args(args, parser):
    # 从命令行中获取参数
    # parser.add_argument('--time_limit', type=int, default=1000, help='Time limit for the environment')
    # parser.add_argument('--key', type=str, default="robotarium_gym:HeterogeneousSensorNetwork-v0",
    #                     help='Key for the environment')

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # 从算法config.py中获取参数
    parser = get_config()
    # 从parse_args命令行中获取参数
    all_args = parse_args(args, parser)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 创建多线程训练环境
    envs, envs_args = make_train_env_robo(all_args.n_rollout_threads, all_args.seed)

    # 对参数config的一些修改
    # 设置训练环境名
    all_args.env_name = 'robotarium'
    # 设置episode长度
    all_args.episode_length = envs_args['time_limit']

    # 打印训练环境的参数
    print("Env Name: ", all_args.env_name)
    print("Scenario: ", envs_args["key"])

    # 获取环境名缩写
    def get_shortcut_name(scenario):
        # Dictionary mapping full scenario names to their shortcuts
        shortcuts = {
            "robotarium_gym:PredatorCapturePrey-v0": "PCP",
            "robotarium_gym:PredatorCapturePreyGNN-v0": "PCPGNN",
            "robotarium_gym:Warehouse-v0": "WH",
            "robotarium_gym:Simple-v0": "SMP",
            "robotarium_gym:ArcticTransport-v0": "AT",
            "robotarium_gym:MaterialTransport-v0": "MT",
            "robotarium_gym:MaterialTransportGNN-v0": "MTGNN",
            "robotarium_gym:HeterogeneousSensorNetwork-v0": "HSN"
        }

        return shortcuts.get(scenario, "Unknown Scenario")

    env_short = get_shortcut_name(envs_args["key"])
    envs_args["env_short"] = env_short

    # 把env dict添加到all_args namespace中
    for key, value in envs_args.items():
        setattr(all_args, key, value)

    # 智能体数量
    num_agents = envs.n_agents

    # 如果是mat_dec算法，设置actor共享参数
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
                       0] + "/results") / all_args.env_name / all_args.env_short / all_args.algorithm_name / all_args.experiment_name
    print("run_dir: ", run_dir)
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    all_args.run_dir = run_dir

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

    # 设置进程名
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.env_short) + \
        "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # 创建多线程测试环境
    all_args.use_eval = False  # 暂时不用测试环境
    eval_envs = make_eval_env_robo(all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])

