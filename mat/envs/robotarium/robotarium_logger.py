import wandb

from mat.algorithms.utils.base_logger import BaseLogger
import time
from functools import reduce
import numpy as np


class RobotariumLogger(BaseLogger):
    def __init__(self, args, num_agents, writter, run_dir):
        super(RobotariumLogger, self).__init__(
            args, num_agents, writter, run_dir
        )

        # declare some variables

        self.total_num_steps = None
        self.episode = None
        self.start = None
        self.episodes = None

        self.n_rollout_threads = self.args.n_rollout_threads

        self.train_episode_rewards = None
        self.one_episode_len = None
        self.done_episode_infos = None
        self.done_episodes_rewards = None
        self.done_episode_lens = None


    def get_task_name(self):
        return f"{self.args.key}"

    def init(self, episodes):
        # 记录训练开始时间
        self.start = time.time()
        # episodes总个数
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(self.n_rollout_threads)
        self.one_episode_len = np.zeros(self.n_rollout_threads, dtype=int)
        self.done_episodes_rewards = np.zeros(self.n_rollout_threads)
        self.done_episode_lens = np.zeros(self.n_rollout_threads)
        self.done_episode_infos = [{} for _ in range(self.n_rollout_threads)]

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        # 当前是第几个episode
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        # 并行环境中的每个环境是否done （n_env_threads, ）
        dones_env = np.all(dones, axis=1)
        # 并行环境中的每个环境的step reward （n_env_threads, ）
        reward_env = np.mean(rewards, axis=1).flatten()
        # 并行环境中的每个环境的episode reward （n_env_threads, ）累积
        self.train_episode_rewards += reward_env
        # 并行环境中的每个环境的episode len （n_env_threads, ）累积
        self.one_episode_len += 1

        for t in range(self.n_rollout_threads):
            # 如果这个环境的episode结束了
            if dones_env[t]:
                # 已经done的episode的总reward
                self.done_episodes_rewards[t] = self.train_episode_rewards[t]
                self.train_episode_rewards[t] = 0  # 归零这个以及done的episode的reward

                # 存一下这个已经done的episode的terminated step的信息
                self.done_episode_infos[t] = infos[t][0]

                # 存一下这个已经done的episode的episode长度
                self.done_episode_lens[t] = self.one_episode_len[t]
                self.one_episode_len[t] = 0  # 归零这个以及done的episode的episode长度

                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[t]['episode_return'] * self.num_agents == \
                       self.done_episodes_rewards[t] or self.done_episode_infos[t]['episode_return'] == \
                       self.done_episodes_rewards[t], 'episode reward not match'
                # 检查环境保存的episode reward和episode len与算法口的信息是否一致
                assert self.done_episode_infos[t]['episode_steps'] == self.done_episode_lens[t], 'episode len not match'

    def episode_log(
            self, train_infos, buffer
    ):
        """Log information for each episode."""

        # 记录训练结束时间
        self.end = time.time()

        # 当前跑了多少time steps
        self.total_num_steps = (
                self.episode
                * self.args.episode_length
                * self.n_rollout_threads
        )

        print(
            "Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args.env_name,
                self.args.algorithm_name,
                self.args.experiment_name,
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.args.num_env_steps,
                int(self.total_num_steps / (self.end - self.start)),
            )
        )

        # 记录每个episode的平均total overlap
        average_total_overlap = np.mean([info["total_overlap"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_total_overlap",
            {"average_total_overlap": average_total_overlap},
            self.total_num_steps,
        )
        # 记录每个episode的平均total reward
        average_total_reward = np.mean([info["episode_return"] for info in self.done_episode_infos])

        # 记录每个episode的平均edge count
        average_edge_count = np.mean([info["edge_count"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_edge_count",
            {"average_edge_count": average_edge_count},
            self.total_num_steps,
        )

        # 记录每个episode的平均violations
        average_violations = np.mean([info["violation_occurred"] for info in self.done_episode_infos])
        self.writter.add_scalars(
            "average_violations",
            {"average_violations": average_violations},
            self.total_num_steps,
        )

        self.done_episode_infos = [{} for _ in range(self.n_rollout_threads)]

        # 记录每个episode的平均episode length
        average_episode_len = (
            np.mean(self.done_episode_lens) if len(self.done_episode_lens) > 0 else 0.0
        )
        self.done_episode_lens = np.zeros(self.n_rollout_threads)

        self.writter.add_scalars(
            "average_episode_length",
            {"average_episode_length": average_episode_len},
            self.total_num_steps,
        )

        # 记录每个episode的平均 step reward
        train_infos["average_step_rewards"] = buffer.get_mean_rewards()
        self.log_train(train_infos, self.total_num_steps)

        self.writter.add_scalars(
            "average_step_rewards",
            {"average_step_rewards": train_infos["average_step_rewards"]},
            self.total_num_steps,
        )
        print(
            "Average step reward is {}.".format(
                train_infos["average_step_rewards"]
            )
        )

        # 记录每个episode的平均 episode reward
        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards)
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.writter.add_scalars(
                "train_episode_rewards",
                {"aver_rewards": aver_episode_rewards},
                self.total_num_steps,
            )
            self.done_episodes_rewards = np.zeros(self.n_rollout_threads)


    def log_train(self, train_infos, total_num_steps):
        for k, v in train_infos.items():
            if self.args.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
