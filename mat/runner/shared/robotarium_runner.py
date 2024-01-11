import time
import wandb
import numpy as np
from functools import reduce
import torch
from mat.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class RobotariumRunner(Runner):
    # 好神奇 不同的env有不同的runner
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(RobotariumRunner, self).__init__(config)

    # def run2(self):
    #     for episode in range(1):
    #         self.eval(episode)

    def run(self):
        # 开始训练
        print("start running")
        # 在环境reset之后返回的obs，share_obs，available_actions存入replay buffer
        self.warmup()

        # 记录训练开始时间
        start = time.time()
        # 计算总共需要跑多少个episode = 总训练时间步数 / 每个episode的时间步数 / 并行的环境数 (int)
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        # 初始化一些logger 临时
        train_episode_rewards = np.zeros(self.n_rollout_threads)
        one_episode_len = np.zeros(self.n_rollout_threads, dtype=int)
        done_episodes_rewards = []
        episode_lens = []
        done_episode_infos = []


        # 开始训练！！！！！！
        # 对于每一个episode
        for episode in range(episodes):
            # 学习率是否随着episode线性递减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            # 对于每一个episode中的每一个step
            for step in range(self.episode_length):
                # Sample actions
                """
                采样动作 - 进入actor network 
                values: (n_threads, n_agents, 1)
                actions: (n_threads, n_agents, action_dim)
                action_log_probs: (n_threads, n_agents, 1)
                rnn_states: (进程数量, n_agents, rnn层数, rnn层大小)
                rnn_states_critic: (n_threads, rnn层数, rnn_hidden_dim)
                """
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)

                """
                在得到动作后，执行动作
                与环境交互一个step，得到obs，share_obs，rewards，dones，infos，available_actions
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads, n_agents)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                """
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                """临时的一些log信息"""
                # 并行环境中的每个环境是否done （n_env_threads, ）
                dones_env = np.all(dones, axis=1)
                # 并行环境中的每个环境的step reward （n_env_threads, ）
                reward_env = np.mean(rewards, axis=1).flatten()
                # 并行环境中的每个环境的episode reward （n_env_threads, ）累积
                train_episode_rewards += reward_env
                # 并行环境中的每个环境的episode len （n_env_threads, ）累积
                one_episode_len += 1

                for t in range(self.n_rollout_threads):
                    # 如果这个环境的episode结束了
                    if dones_env[t]:
                        # 已经done的episode的总reward
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward

                        # 存一下这个已经done的episode的terminated step的信息
                        done_episode_infos.append(infos[t][0])

                        # 存一下这个已经done的episode的episode长度
                        episode_lens.append(one_episode_len[t].copy())
                        one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度

                        # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                        # if not self.done_episode_infos[t]['episode_return'] * self.env_args['n_agents'] == \
                        #        self.done_episodes_rewards[t]:
                        #     print('stop here')
                        assert done_episode_infos[t]['episode_return'] * self.num_agents == \
                               done_episodes_rewards[t], 'episode reward not match'
                        # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                        assert done_episode_infos[t]['episode_steps'] == episode_lens[
                            t], 'episode len not match'

                    # insert data into buffer
                    """把这一步的数据存入replay buffer"""
                    self.insert(data)

            # 收集完了一个episode的所有timestep data，开始计算return
            # compute Q and V using GAE
            self.compute()

            # train
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # 记录每个episode的平均total overlap
                average_total_overlap = np.mean([info["total_overlap"] for info in done_episode_infos])
                self.writter.add_scalars(
                    "average_total_overlap",
                    {"average_total_overlap": average_total_overlap},
                    total_num_steps,
                )
                # 记录每个episode的平均total reward
                average_total_reward = np.mean([info["episode_return"] for info in done_episode_infos])

                # 记录每个episode的平均edge count
                average_edge_count = np.mean([info["edge_count"] for info in done_episode_infos])
                self.writter.add_scalars(
                    "average_edge_count",
                    {"average_edge_count": average_edge_count},
                    total_num_steps,
                )

                # 记录每个episode的平均violations
                average_violations = np.mean([info["violation_occurred"] for info in done_episode_infos])
                self.writter.add_scalars(
                    "average_violations",
                    {"average_violations": average_violations},
                    total_num_steps,
                )
                done_episode_infos = []

                # 记录每个episode的平均episode length
                average_episode_len = (
                    np.mean(episode_lens) if len(episode_lens) > 0 else 0.0
                )
                episode_lens = []
                self.writter.add_scalars(
                    "average_episode_length",
                    {"average_episode_length": average_episode_len},
                    total_num_steps,
                )

                # 记录每个episode的平均 episode reward
                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print(
                        "Some episodes done, average episode reward is {}.\n".format(
                            aver_episode_rewards
                        )
                    )
                    self.writter.add_scalars(
                        "train_episode_rewards",
                        {"aver_rewards": aver_episode_rewards},
                        total_num_steps,
                    )
                    done_episodes_rewards = []

                print("Episode return: ", aver_episode_rewards)

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        """
        Warm up the replay buffer.
        在环境reset之后返回的obs，share_obs，available_actions存入replay buffer
        """
        # reset env
        """
        obs: (n_threads, n_agents, obs_dim)
        share_obs: (n_threads, n_agents, share_obs_dim)
        available_actions: (n_threads, n_agents, action_dim)
        """
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        # 在环境reset之后，把所有并行环境下的每一个agent在t=0时的obs,share_obs,available_actions放入buffer里的self.里
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def collect(self, step):
        # 把trainer网络都切换到eval模式
        self.trainer.prep_rollout()

        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              np.concatenate(self.buffer.available_actions[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_state_critic), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [
                [
                    [0.0]
                    if "bad_transition" in info[agent_id].keys()
                       and info[agent_id]['bad_transition'] == True
                    else [1.0]
                    for agent_id in range(self.num_agents)
                ]
                for info in infos
            ]
        )

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks, bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        pass
