from mat.envs.env_wrappers import ShareSubprocVecEnv_robotarium, ShareSubprocVecEnv
from mat.envs import REGISTRY as env_REGISTRY
from functools import partial
from multiprocessing import Pipe, Process
import yaml



def make_train_env_robo(n_rollout_threads, seed):
    # 读取gymma里面的env参数
    with open('/home/tsaisplus/MuRPE_base/Multi-Agent-Transformer/mat/envs/robotarium/gymma.yaml') as stream:
        gymma_args = yaml.safe_load(stream)

    env_fn = env_REGISTRY[gymma_args["env"]]

    # 设置环境的随机种子
    gymma_args["env_args"]["seed"] = seed
    # merge two dicts
    env_args = gymma_args["env_args"]

    # 从args中单独提取出env_args，生成多进程的env_args
    env_args = [env_args.copy() for _ in range(n_rollout_threads)]


    # 对于每一个进程，都将env_args中的seed加上一个偏移量
    for i in range(n_rollout_threads):
        env_args[i]["seed"] += i

    if n_rollout_threads > 1:
        return ShareSubprocVecEnv_robotarium(env_fn, env_args)
    else:
        raise NotImplementedError("The number of rollout threads must be greater than 1.")


def make_eval_env_robo(all_args):
    raise NotImplementedError
