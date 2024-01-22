import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state):
        super(Encoder, self).__init__()

        # 无效的state_dim
        self.state_dim = state_dim
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        # 是否额外encode state
        self.encode_state = encode_state

        # state_encoder和obs_encoder都是单层的MLP
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)

        # 不同的地方！！GRU in stead of self attention encoder block
        # self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.gru = nn.GRU(n_embd, n_embd, num_layers=2, batch_first=True)

        # 最后的head是一个MLP，输出的是一个值 V
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # obs: (n_rollout_thread, n_agent, obs_dim)
        if self.encode_state:
            assert self.encode_state == False
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            # 所有agent共用同一个obs_encoder，这是第一个embedding
            # 分别提取每个agent的obs feature
            obs_embeddings = self.obs_encoder(obs)
            # obs_embeddings: (n_rollout_thread, n_agent, n_embd)
            x = obs_embeddings

        # 在做完layer norm之后，进入GRU
        # rep: (n_rollout_thread, n_agent, n_embd)
        rep, _ = self.gru(self.ln(x))
        # v_loc: (n_rollout_thread, n_agent, 1)
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type == 'Discrete':
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                nn.GELU())
        else:
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
            self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
        self.ln = nn.LayerNorm(n_embd)
        # self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.gru = nn.GRU(n_embd, n_embd, num_layers=2, batch_first=True)
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        action_embeddings = self.action_encoder(action)
        x = action_embeddings
        x += obs_rep
        x, _ = self.gru(self.ln(x))
        logit = self.head(x)

        return logit


class MultiAgentGRU(nn.Module):
    # 和MultiAgentTransformer一模一样
    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentGRU, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        # 初始化encoder和decoder
        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        # 检查type和device
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        # 给encoder输入state和obs: [episode_length, num_agents, obs_dim]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)
        # [episode_length, num_agents, 1]
        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        """
        和transformer_policy里面的get_actions对应
        输入当前timestep的share_obs, obs, available_actions
        输出当前timestep的actions, action_log_probs, values

        state: [n_rollout_threads, num_agents, share_obs_dim]
        obs: [n_rollout_threads, num_agents, obs_dim]
        available_actions: [n_rollout_threads, num_agents, action_dim]
        """
        # 获取obs的维度
        ori_shape = np.shape(obs)
        # 这个state没用到 [n_rollout_threads, num_agents, 37]
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        # 检查type和device
        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # batch_size == n_rollout_threads 并行环境数量
        batch_size = np.shape(obs)[0]

        # 给encoder输入state和obs: [n_rollout_threads, num_agents, obs_dim]
        v_loc, obs_rep = self.encoder(state, obs)
        # 输出v_loc [n_rollout_threads, num_agents, 1] --> 相当于每个agent都有一个v_loc
        # obs_rep [n_rollout_threads, num_agents, n_embd]

        if self.action_type == "Discrete":
            # (n_rollout_threads, n_agent, 1)
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)
        # (n_rollout_threads, n_agent, 1)
        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        # v_loc: (n_rollout_thread, n_agent, 1)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot



