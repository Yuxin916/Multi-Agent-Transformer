import torch
import numpy as np
import torch.nn.functional as F
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])


def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols


class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information. 所有参数
    :param num_agents: (int) number of agents in the env. 环境中的agent数量
    :param obs_space: (gym.Space) observation space of agents. 单个agent观测空间
    :param cent_obs_space: (gym.Space) centralized observation space of agents. 单个agent的共享观测空间
    :param act_space: (gym.Space) action space for agents. 单个agent的动作空间
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, env_name):
        # args
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.env_name = env_name

        # 单个agent的观测空间维度和共享观测空间维度
        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        """
        Buffer里储存了：ALL (np.ndarray) 相当于把agent都存到一起了
        1. self.share_obs: 全局状态 [episode_length + 1, 进程数量, agent数量, 全局状态维度]
        2. self.obs: 局部状态 [episode_length + 1, 进程数量, agent数量, 局部状态维度]
        3. self.rnn_states: actor网络的RNN状态 [episode_length + 1, 进程数量, agent数量, recurrent_N, hidden_size]
        4. self.rnn_states_critic: critic网络的RNN状态 [episode_length + 1, 进程数量, agent数量, recurrent_N, hidden_size]
        5. self.value_preds: critic的value预测 [episode_length + 1, 进程数量, agent数量, 1]
        6. self.returns: [episode_length + 1, 进程数量, agent数量, 1]
        7. self.advantages: [episode_length, 进程数量, agent数量, 1]
        8. self.available_actions: [episode_length + 1, 进程数量, agent数量, 动作维度]
        9. self.actions: [episode_length, 进程数量, agent数量, 动作维度]
        10. self.action_log_probs: [episode_length, 进程数量, agent数量, 动作维度]
        11. self.rewards: [episode_length, 进程数量, agent数量, 1]
        12. self.masks: [episode_length + 1, 进程数量, agent数量, 1]
        13. self.bad_masks: [episode_length + 1, 进程数量, agent数量, 1]
        14. self.active_masks: [episode_length + 1, 进程数量, agent数量, 1]
        15. info
        """
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)

        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.returns = np.zeros_like(self.value_preds)

        self.advantages = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)

        self.action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, act_shape),
                                         dtype=np.float32)

        self.rewards = np.zeros((self.episode_length, self.n_rollout_threads, num_agents, 1),
                                dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def get_mean_rewards(self):
        """Get mean rewards for logging."""
        return np.mean(self.rewards)

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
            next_value: (np.ndarray) value predictions for the step after the last episode step.
            # V（s_T+1） shape=(环境数, agent数量，1)
            value_normalizer: (ValueNorm) If not None, ValueNorm value normalizer instance.
            # self.value_normalizer --- ValueNorm

        在下面的计算过程中
            输入： next_value [batch_size, num_agents, 1]
            最后输出：
                self.gae [episode_length + 1, thread, num_agents, 1]
                self.returns [episode_length + 1， thread, num_agents, 1] 这个episode每一步的每个人的Q值
                self.value_preds [episode_length + 1, thread, num_agents, 1] 这个episode每一步的每个人的V值
                gae = Q - V
        """

        # 把最后一个状态的状态值放到value_preds的最后一个位置, index是200
        self.value_preds[-1] = next_value
        # 可以看成gae(200) = 0
        gae = 0
        # timestep从后往前--倒推的方式 # 从step199到0
        for step in reversed(range(self.rewards.shape[0])):
            # use ValueNorm
            # 在GAE计算中，将值函数的估计值denormalize，然后再计算GAE，最后再normalize
            if self._use_popart or self._use_valuenorm:
                # 计算delta[step]
                # delta[step] = r([step]) + gamma * V(s[step+1]) * mask - V(s[step]) -- 如果下一个step 不done
                # delta[step] = r([step]) + gamma * 0 * mask - V(s[step]) -- 如果下一个step done
                delta = (  # t时刻的delta
                    self.rewards[step]
                    + self.gamma
                    * value_normalizer.denormalize(self.value_preds[step + 1])   # 在计算delta的时候denormalize
                    * self.masks[step + 1]   # 如果下一个step 不done, self.value_preds[step + 1]才存在
                    - value_normalizer.denormalize(self.value_preds[step]) # 在计算delta的时候denormalize
                        )

                # gae递归公式，查看https://zhuanlan.zhihu.com/p/651944382和笔记
                # gae[step] = delta[step] + gamma * lambda * mask[t+1] * gae[t+1]
                gae = ( # 根据t+1时刻的gae计算t时刻的gae
                    delta
                    +
                    self.gamma * self.gae_lambda
                    * self.masks[step + 1]
                    * gae  # gae在for loop里面迭代, 这个代表的是t+1时刻的gae
                )

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                # 保存gae[step]到self
                self.advantages[step] = gae
                # Q -- V网络的标签值 = GAE(step) + V网络(step) -- 标量
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                pass
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]
        pass

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # batchsize是环境数量*episode长度
        batch_size = n_rollout_threads * episode_length

        # 产生mini_batch大小
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        # 随机打乱0到batch_size-1
        rand = torch.randperm(batch_size).numpy()
        # 把rand分成num_mini_batch份，每份mini_batch_size大小
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        """
        原始的
        1. self.share_obs: 全局状态 [episode_length + 1, 进程数量, agent数量, 全局状态维度]
        2. self.obs: 局部状态 [episode_length + 1, 进程数量, agent数量, 局部状态维度]
        3. self.rnn_states: actor网络的RNN状态 [episode_length + 1, 进程数量, agent数量, recurrent_N, hidden_size]
        4. self.rnn_states_critic: critic网络的RNN状态 [episode_length + 1, 进程数量, agent数量, recurrent_N, hidden_size]
        5. self.value_preds: critic的value预测 [episode_length + 1, 进程数量, agent数量, 1]
        6. self.returns: [episode_length + 1, 进程数量, agent数量, 1]
        7. self.advantages: [episode_length, 进程数量, agent数量, 1]
        8. self.available_actions: [episode_length + 1, 进程数量, agent数量, 动作维度]
        9. self.actions: [episode_length, 进程数量, agent数量, 动作维度]
        10. self.action_log_probs: [episode_length, 进程数量, agent数量, 动作维度]
        11. self.rewards: [episode_length, 进程数量, agent数量, 1]
        12. self.masks: [episode_length + 1, 进程数量, agent数量, 1]
        13. self.bad_masks: [episode_length + 1, 进程数量, agent数量, 1]
        14. self.active_masks: [episode_length + 1, 进程数量, agent数量, 1]
        新的
        把episode_length和进程数量合并到一起 （batchsize），然后是agent数量和obs的维度
        """
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        # 把batchsize和agent数量合并到一起 200,3,1 -》 600,1
        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
