import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from mat.utils.shared_buffer import SharedReplayBuffer
from mat.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
from mat.utils.util import save_config
from mat.envs import LOGGER_REGISTRY

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # 如果不使用centralized V，那么obs和share_obs是一样的
        # 使用centralized V, MAPPO
        # 不使用centralized V，IPPO
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V \
            else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        # 检查gymma里的time_limit设置是否正确
        assert self.envs.env_info["episode_limit"] == self.all_args.episode_length, 'episode_limit != episode_length'

        # 保存参数
        save_config(self.all_args, self.run_dir)

        # policy network - TransformerPolicy
        # 初始化了MAT的encoder和decoder
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.envs.action_space[0],
                             self.num_agents,
                             device=self.device)

        # 如果有模型，加载模型
        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm - MATTrainer - loss函数更新
        self.trainer = TrainAlgo(self.all_args,
                                 self.policy,
                                 self.num_agents,
                                 device=self.device)
        
        # buffer - SharedReplayBuffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0],
                                         share_observation_space,
                                         self.envs.action_space[0],
                                         self.all_args.env_name)
        # 环境的logger
        self.logger = LOGGER_REGISTRY[self.all_args.key](self.all_args,
                                                         self.num_agents,
                                                         self.writter,
                                                         self.run_dir)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """
        Compute returns and advantages for collected data.
        训练开始之前，首先调用self.compute()函数计算这个episode的折扣回报
        在计算折扣回报之前，先算这个episode最后一个状态的状态值函数next_values，然后调用compute_returns函数计算折扣回报
        Compute critic evaluation of the last state, V（s-T）
        and then let buffer compute returns, which will be used during training.
        """
        # 把policy网络都切换到eval模式
        self.trainer.prep_rollout()

        # 计算critic的最后一个state的值
        if self.buffer.available_actions is None:
            # next_values： [n_rollout_threads, num_agents, 1] -- 最后一个状态的状态值
            # 每一个agent根据自己的observation进入encoder计算出一个状态值
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            # TransformerPolicy.get_values
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))

        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        # 通过每个agent的最后一个状态的状态值计算折扣回报 GAE (每一步每个agent分别的Q，V，A)
        # next_value --- np.array shape=(环境数, num_agents, 1) -- 最后一个状态的状态值
        # self.value_normalizer --- ValueNorm
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        # 把actor和critic网络都切换回train模式
        self.trainer.prep_training()

        # 从这里开始，mat反向更新
        train_infos = self.trainer.train(self.buffer)

        # 把上一个episode产生的最后一个timestep的state放入buffer的新的episode的第一个timestep
        self.buffer.after_update()

        return train_infos

    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode)

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_dir)
 
    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
