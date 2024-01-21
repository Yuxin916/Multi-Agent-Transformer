from mat.envs.robotarium.robotarium_logger import RobotariumLogger
import time
import numpy as np


class PCPLogger(RobotariumLogger):
    """
    每个环境scenario只有episode_log不一样
    """

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

        # 检查哪个环境done了
        a = [index for index, info in enumerate(self.done_episode_infos) if
                                       "episode_return" in info]
        b = [index for index, value in enumerate(self.done_episode_lens) if value != 0]
        c = [index for index, value in enumerate(self.done_episodes_rewards) if value != 0]
        assert a == b == c
        indices = a


        # # 记录每个episode的平均total overlap
        # average_total_overlap = np.mean([info["total_overlap"] for info in self.done_episode_infos])
        # self.writter.add_scalars(
        #     "average_total_overlap",
        #     {"average_total_overlap": average_total_overlap},
        #     self.total_num_steps,
        # )
        # 记录每个episode的平均total reward 和 total step
        episode_returns = [self.done_episode_infos[index]["episode_return"] for index in indices]
        episode_step = [self.done_episode_infos[index]["episode_steps"] for index in indices]

        # 记录每个episode的平均avergae reward 和 average step
        average_episode_return = np.mean(episode_returns) if episode_returns else 0
        average_episode_step = np.mean(episode_step) if episode_step else 0

        self.writter.add_scalars(
            "average_episode_length",
            {"average_episode_length": average_episode_step},
            self.total_num_steps,
        )
        print(
            "Some episodes done, average episode length is {}.\n".format(
                average_episode_step
            )
        )

        print(
            "Some episodes done, average episode reward is {}.\n".format(
                average_episode_return*self.num_agents
            )
        )
        self.writter.add_scalars(
            "train_episode_rewards",
            {"aver_rewards": average_episode_return*self.num_agents},
            self.total_num_steps,
        )

        for index in indices:
            self.done_episode_infos[index] = {}
            self.done_episode_lens[index] = 0
            self.done_episodes_rewards[index] = 0

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

