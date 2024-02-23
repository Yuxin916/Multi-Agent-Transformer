import torch
import numpy as np
from mat.utils.util import update_linear_schedule
from mat.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from mat.algorithms.utils.util import check
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer


class TransformerPolicy:
    """
    MAT Policy class. MAT的encoder和decoder
    计算actions和value function predictions （actor和critic）

    :param args: (argparse.Namespace) arguments containing relevant model and policy information. 所有参数
    :param obs_space: (gym.Space) observation space. 单个agent观测空间
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
                            单个agent的共享观测空间
    :param act_space: (gym.Space) action space. 单个agent动作空间
    :param device: (torch.device) specifies the device to run on (cpu/gpu). 设备
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.algorithm_name = args.algorithm_name
        print("algorithm_name: ", self.algorithm_name)
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self._use_policy_active_masks = args.use_policy_active_masks
        # 判断动作空间是连续还是离散
        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        # 单个agent的观测空间维度和共享观测空间维度
        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_obs_dim = get_shape_from_obs_space(cent_obs_space)[0]

        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        # agent数量
        self.num_agents = num_agents
        print("num_agents: ", self.num_agents)

        self.tpdv = dict(dtype=torch.float32, device=device)

        # 根据算法名字选择不同的模型
        if self.algorithm_name in ["mat", "mat_dec"]:
            from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer as MAT
        elif self.algorithm_name == "mat_gru":
            from mat.algorithms.mat.algorithm.mat_gru import MultiAgentGRU as MAT
        elif self.algorithm_name == "mat_decoder":
            from mat.algorithms.mat.algorithm.mat_decoder import MultiAgentDecoder as MAT
        elif self.algorithm_name == "mat_encoder":
            from mat.algorithms.mat.algorithm.mat_encoder import MultiAgentEncoder as MAT
        elif self.algorithm_name == "mat_gat":
            from mat.algorithms.mat.algorithm.mat_gat import MultiAgentTransformer_GAT as MAT
        else:
            raise NotImplementedError

        # 初始化模型
        # 使用config里的“add for transformer”参数以及连续/离散动作类型
        # self.share_obs_dim没有使用
        self.transformer = MAT(self.share_obs_dim, self.obs_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, device=device,
                               action_type=self.action_type, dec_actor=args.dec_actor,
                               share_actor=args.share_actor)

        if args.env_name == "hands":
            self.transformer.zero_std()

        # count the volume of parameters of model
        # 计算模型参数量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.transformer.parameters():
            mulValue = np.prod(param.size())
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

        # 优化器
        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr,
                                          eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        采样动作 - 进入actor network获取动作和动作的log_prob，以及critic network的value

        输入当前timestep的share_obs, obs, available_actions
        输出当前timestep的actions, action_log_probs, values

        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """

        # reshape一下cent_obs和obs和available_actions

        # [n_rollout_threads, num_agents, share_obs_dim]
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        # [n_rollout_threads, num_agents, obs_dim]
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            # [n_rollout_threads, num_agents, act_dim]
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        # 进入MAT模型获取actions，策略概率以及每一个agent的value
        # (n_rollout_threads, n_agent, 1)
        actions, action_log_probs, values = self.transformer.get_actions(cent_obs,
                                                                         obs,
                                                                         available_actions,
                                                                         deterministic)
        # reshape一下actions, action_log_probs, values
        # [n_rollout_threads * num_agents, 1]
        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        # unused, just for compatibility
        rnn_states_actor = check(rnn_states_actor).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, obs, rnn_states_critic, masks, available_actions=None):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        # RESHAPE
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)
        """
        cent_obs: [n_rollout_threads, num_agents, share_obs_dim]
        obs: [n_rollout_threads, num_agents, obs_dim]
        available_actions: [n_rollout_threads, num_agents, act_dim]
        """
        # [n_rollout_threads, num_agents, 1]
        values = self.transformer.get_values(cent_obs, obs, available_actions)

        values = values.view(-1, 1)

        # (n_rollout_threads * num_agents, 1)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param actions: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        # reshape一下cent_obs和obs和available_actions -》 [episode, num_agents, share_obs_dim]
        cent_obs = cent_obs.reshape(-1, self.num_agents, self.share_obs_dim)
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        actions = actions.reshape(-1, self.num_agents, self.act_num)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        # (episode_length, n_agent, 1)
        action_log_probs, values, entropy = self.transformer(cent_obs, obs, actions, available_actions)

        # reshape to [episode_length * num_agents, 1]
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)
        entropy = entropy.view(-1, self.act_num)

        if self._use_policy_active_masks and active_masks is not None:
            entropy = (entropy*active_masks).sum()/active_masks.sum()
        else:
            entropy = entropy.mean()

        return values, action_log_probs, entropy

    def act(self, cent_obs, obs, rnn_states_actor, masks, available_actions=None, deterministic=True):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """

        # this function is just a wrapper for compatibility
        rnn_states_critic = np.zeros_like(rnn_states_actor)
        _, actions, _, rnn_states_actor, _ = self.get_actions(cent_obs,
                                                              obs,
                                                              rnn_states_actor,
                                                              rnn_states_critic,
                                                              masks,
                                                              available_actions,
                                                              deterministic)

        return actions, rnn_states_actor

    def save(self, save_dir, episode):
        torch.save(self.transformer.state_dict(), str(save_dir) + "/transformer_" + str(episode) + ".pt")

    def restore(self, model_dir):
        transformer_state_dict = torch.load(model_dir)
        self.transformer.load_state_dict(transformer_state_dict)
        # self.transformer.reset_std()

    def train(self):
        self.transformer.train()

    def eval(self):
        # 把policy网络都切换到eval模式
        self.transformer.eval()

