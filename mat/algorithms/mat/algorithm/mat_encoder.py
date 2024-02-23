import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical, Normal
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()
        # d_q = d_k = d_v = d_model/h
        # d_model = n_embd --> 确保d_model可以被n_head整除
        assert n_embd % n_head == 0
        # 使用mask, 使得attention只能看到前面的信息
        self.masked = masked
        # 多头
        self.n_head = n_head
        # key, query, value 线性projection for all heads (进入attention前的线性变换)
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # 从多头attention输出再concat后的线性变换
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # 生成一个下三角矩阵，用于mask
        # dimension: (1, 1, n_agent + 1, n_agent + 1)
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        """
        输入: key, value, query
        计算: scaled dot-product self-attention
        输出: self-attention后的value
        """
        # B: batch size - n_rollout_threads
        # L: sequence length - n_agents
        # D: hidden dimension - n_embd
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # key, query, value分别进入线性变换，然后reshape成(B, head数量, L, n_embd//head数[dimension of each head])的形式
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # Attention（Q,K,V）= softmax(QK^T/sqrt(d_k))V

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            # mask out L 之后的信息 也就是只能看到之前的agent的action
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)

        # 把multi-head的结果拼接起来 --> (B, L, D)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection --> (B, L, D)
        y = self.proj(y)

        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # mask关掉了，说明所有的agent都可以看到所有的agent的obs
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        # x [n_rollout_thread, n_agents, n_embd]
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_block, n_embd,
                 n_head, n_agent, encode_state, action_type='Discrete'):
        super(Encoder, self).__init__()

        # 无效的state_dim
        self.state_dim = state_dim
        # 单个智能体obs_dim
        self.obs_dim = obs_dim
        # 单个智能体action_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        # 是否额外encode state
        self.encode_state = encode_state
        self.action_type = action_type

        # state_encoder和obs_encoder都是单层的MLP
        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)

        # n_block代表EncodeBlock的数量
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

        # 最后的head是一个MLP，输出的是一个值 V
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))
        # 最后的act_head是一个MLP，输出的是action_dim的logits
        self.act_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, action_dim)))
        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    def forward(self, state, obs):
        # state: (n_rollout_thread, n_agent, state_dim)
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

        # 在做完layer norm之后，进入multi-head attention, 每一个都是EncodeBlock
        # rep: (n_rollout_thread, n_agent, n_embd)
        rep = self.blocks(self.ln(x))

        # v_loc: (n_rollout_thread, n_agent, 1)
        v_loc = self.head(rep)

        # logit: (n_rollout_thread, n_agent, action_dim)
        logit = self.act_head(rep)

        return v_loc, rep, logit


class MultiAgentEncoder(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(MultiAgentEncoder, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        # 初始化encoder，这里的encoder包含MLP决策
        self.encoder = Encoder(state_dim, obs_dim, action_dim, n_block, n_embd, n_head, n_agent, encode_state,
                               action_type=self.action_type)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.encoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        """

        state: (batch, n_agent, state_dim)
        obs: (batch, n_agent, obs_dim)
        action: (batch, n_agent, 1)
        available_actions: (batch, n_agent, act_dim)
        """

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
        # 输出value, obs_rep和action的logit
        v_loc, obs_rep, logit = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            # logit: (batch, n_agent, action_dim)
            if available_actions is not None:
                logit[available_actions == 0] = -1e10
            # 注意这里不分agent，直接把所有agent的动作都放到一起了
            distri = Categorical(logits=logit)
            # action/action_log/entropy: (batch, n_agent, 1)
            action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
            entropy = distri.entropy().unsqueeze(-1)
        else:
            act_mean = logit
            action_std = torch.sigmoid(self.encoder.log_std) * 0.5
            distri = Normal(act_mean, action_std)
            action_log = distri.log_prob(action)
            entropy = distri.entropy()

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
        # 给encoder输入state和obs: [n_rollout_threads, num_agents, obs_dim]
        v_loc, obs_rep, logit = self.encoder(state, obs)
        # 输出v_loc [n_rollout_threads, num_agents, 1] --> 相当于每个agent都有一个v_loc
        # obs_rep [n_rollout_threads, num_agents, n_embd]
        # logits [n_rollout_threads, num_agents, action dim]
        if self.action_type == "Discrete":
            if available_actions is not None:
                logit[available_actions == 0] = -1e10

            distri = Categorical(logits=logit)
            output_action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            output_action_log = distri.log_prob(output_action)
            output_action = output_action.unsqueeze(-1)
            output_action_log = output_action_log.unsqueeze(-1)
        else:
            act_mean = logit
            action_std = torch.sigmoid(self.encoder.log_std) * 0.5
            distri = Normal(act_mean, action_std)
            output_action = act_mean if deterministic else distri.sample()
            output_action_log = distri.log_prob(output_action)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        # v_loc: (n_rollout_thread, n_agent, 1)
        v_tot, _, _ = self.encoder(state, obs)
        return v_tot



