import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F


def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False):
    # 相比于output_action多出来一个a0来trigger decoder,
    # 为什么是给action dim + 1呢？ 而不是给N+1呢？
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    # 我想输出的是一个(batch, n_agent, 1)的action
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    # 对于每一个agent来说，autoregressive的生成动作
    for i in range(n_agent):
        # 对于agent i来说，他的动作是由前面已经完成动作的agent的动作和masked obs来决定的

        # logit: (batch, action_dim) 只第i个agent的动作
        # 进入ma_transformer里decoder的forward pass
        logit = decoder(shifted_action, obs_rep, obs)[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        # agent i's policy distribution
        # distri -- (batch, action_dim)
        distri = Categorical(logits=logit)

        # 根据策略分布选取action (batch,)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        # 被选取action的probability action_log (batch,)
        action_log = distri.log_prob(action)

        # 把第i个agent的动作加入到output_action中
        output_action[:, i, :] = action.unsqueeze(-1)
        # 把第i个agent的动作的log probability加入到output_action_log中
        output_action_log[:, i, :] = action_log.unsqueeze(-1)

        # 把第i个agent的动作加入到shifted_action中，为下一个agent的动作决策做准备
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)

    # (batch, n_agent, 1)
    return output_action, output_action_log


def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    # logit: (batch, n_agent, action_dim)
    logit = decoder(shifted_action, obs_rep, obs)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    # 注意这里不分agent，直接把所有agent的动作都放到一起了
    distri = Categorical(logits=logit)
    # action/action_log/entropy: (batch, n_agent, 1)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
