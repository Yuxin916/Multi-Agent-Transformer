
import socket
from absl import flags
from functools import partial
from mat.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from mat.envs.robotarium.multiagentenv import MultiAgentEnv
from mat.envs.robotarium.gymmawrapper import _GymmaWrapper
import sys
import os
from mat.envs.robotarium.robotarium_gym.scenarios.HeterogeneousSensorNetwork.hsn_logger import HSNLogger
from mat.envs.robotarium.robotarium_gym.scenarios.PredatorCapturePrey.pcp_logger import PCPLogger
# from mat.envs.robotarium.robotarium_gym.scenarios.Warehouse.wh_logger import WHLogger
from mat.envs.robotarium.robotarium_gym.scenarios.ArcticTransport.at_logger import ATLogger
from mat.envs.robotarium.robotarium_gym.scenarios.MaterialTransport.mt_logger import MTLogger


# this function builds a class env (type MultiAgentEnv) with keyworld argument **kwargs.
def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_fn函数的输入是env和**kwargs，输出是MultiAgentEnv类的对象env(**kwargs)
    # 进入_GymmaWrapper类的__init__函数
    return env(**kwargs)

LOGGER_REGISTRY = {
    "robotarium_gym:HeterogeneousSensorNetwork-v0": HSNLogger,
    "robotarium_gym:PredatorCapturePrey-v0": PCPLogger,
    # "robotarium_gym:PredatorCapturePreyGNN-v0"
    # "robotarium_gym:Warehouse-v0": WHLogger,
    # "robotarium_gym:Simple-v0"
    "robotarium_gym:ArcticTransport-v0": ATLogger,
    "robotarium_gym:MaterialTransport-v0": MTLogger,
    # "robotarium_gym:MaterialTransportGNN-v0"
}


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


