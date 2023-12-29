import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
sys.path.append("../../")
from multiprocessing import Pipe, Process
from mat.config import get_config
from mat.envs import REGISTRY as env_REGISTRY
from functools import partial
import yaml
from types import SimpleNamespace as SN
from mat.envs.env_wrappers import ShareSubprocVecEnv_robotarium
from mat.runner.shared.robotarium_runner import RobotariumRunner as Runner


"""Train script for Robotarium."""
def make_train_env_2(all_args):
    # 从args中提取出env的名字 gymmma
    env_fn = env_REGISTRY[all_args.env]

    # 设置环境的随机种子 - 是从算法args中提取出来的
    all_args.env_args["seed"] = all_args.seed
    # 从args中单独提取出env_args，生成多进程的env_args
    env_args = [all_args.env_args.copy() for _ in range(all_args.n_rollout_threads)]
    # 对于每一个进程，都将env_args中的seed加上一个偏移量
    for i in range(all_args.n_rollout_threads):
        env_args[i]["seed"] += i

    assert all_args.n_rollout_threads > 1, "The number of rollout threads for robotarium must be greater than 1."

    return ShareSubprocVecEnv_robotarium(env_fn, env_args, all_args.n_rollout_threads)

def make_eval_env(all_args):
    raise NotImplementedError


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
    config = SN(**vars(parser.parse_args()))
    # 读取gymma里面的env参数
    with open('/home/tsaisplus/MuRPE_base/Multi-Agent-Transformer/mat/envs/robotarium/gymma.yaml') as stream:
        gymma_args = yaml.safe_load(stream)
    gymma_args = SN(**gymma_args)
    # combine the two namespaces
    all_args = SN(**vars(config), **vars(gymma_args))

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
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
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

    # 创建多线程训练环境
    # envs = make_train_env(all_args)
    envs = make_train_env_2(all_args)

    # 创建多线程测试环境
    all_args.use_eval = False # 暂时不用测试环境
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    # env
    all_args.run_dir = run_dir
    num_agents = envs.n_agents

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
    pass

if __name__ == "__main__":
    main(sys.argv[1:])