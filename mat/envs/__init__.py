
import socket
from absl import flags
from functools import partial
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from mat.envs.robotarium.multiagentenv import MultiAgentEnv
from mat.envs.robotarium.gymmawrapper import _GymmaWrapper
import sys
import os



# this function builds a class env (type MultiAgentEnv) with keyworld argument **kwargs.
def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_fn函数的输入是env和**kwargs，输出是MultiAgentEnv类的对象env(**kwargs)
    # 进入_GymmaWrapper类的__init__函数
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

# 新环境的注册
REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)

FLAGS = flags.FLAGS
FLAGS(['train_sc.py'])


