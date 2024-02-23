import copy

import yaml
from gym import spaces

# This file should stay as is when copied to robotarium_eval but local imports must be changed to work with training!
from robotarium_gym.scenarios.PredatorCapturePreyGNN.agent import Agent
from robotarium_gym.scenarios.PredatorCapturePreyGNN.visualize import *
from robotarium_gym.scenarios.base import BaseEnv
from robotarium_gym.scenarios.check_dim import *
from robotarium_gym.utilities.misc import *
from robotarium_gym.utilities.roboEnv import roboEnv
from robotarium_python_simulator.rps.utilities.graph import *
from robotarium_gym.scenarios.check_dim import *


class PredatorCapturePreyGNN(BaseEnv):
    def __init__(self, args):
        # 环境相关的args - /mat/envs/robotarium/robotarium_gym/scenarios/HeterogeneousSensorNetwork/config.yaml
        self.args = args

        # 当前scenario的文件夹
        module_dir = os.path.dirname(__file__)

        # 是否使用随机种子[-1不随机][任意其他数字 随机]
        if self.args.seed != -1:
            np.random.seed(self.args.seed)

        # 从预生成的coalition中加载，文件里包含所有robot的id和radius
        with open(f'{module_dir}/predefined_coalitions.yaml', 'r') as stream:
            self.predefined_coalition = yaml.safe_load(stream)

        # n个agent
        self.num_robots = args.n_agents
        self.num_prey = args.num_prey

        # declare
        self.agent_poses = None  # robotarium convention poses
        self.prey_loc = None
        self.episode_number = 0
        self.action_id2w = {0: 'left', 1: 'right', 2: 'up', 3: 'down', 4: 'no_action'}
        self.action_w2id = {v: k for k, v in self.action_id2w.items()}

        # 初始化所有agent
        self.agents = []
        self.agents = self.load_agents()

        # 单个agent的观测空间dim
        if self.args.capability_aware:
            # [agent_x_pos, agent_y_pos, sensed_prey_x_pose , sensed_prey_y_pose, sensing_radius, capture_radius]
            # Returns the closest prey if multiple agents in range
            self.agent_obs_dim = 6
        elif self.args.agent_id:
            # agent ids are one hot encoded
            # [agent_x_pos, agent_y_pos, sensed_prey_x_pose , sensed_prey_y_pose, sensing_radius, capture_radius, id]
            self.agent_obs_dim = 7
        else:
            # [agent_x_pos, agent_y_pos, sensed_prey_x_pose , sensed_prey_y_pose]
            # Returns the closest prey if multiple agents in range
            self.agent_obs_dim = 4

        # 单个agent的state空间dim = 单个agent的观测空间dim * agent的数量 (假设全局观测 | 所有agent的观测都是一样dim的
        self.agent_state_dim = self.agent_obs_dim * self.args.n_agents

        # 状态空间和动作空间
        actions = []
        observations = []
        states = []

        # 对于每个agent
        for i in range(self.num_robots):
            # 动作空间
            actions.append(spaces.Discrete(5))

            # The lowest any observation will be is -5 (prey loc when can't see one),
            # the highest is 3 (largest reasonable radius an agent will have)
            observations.append(spaces.Box(low=-5, high=3, shape=(self.agent_obs_dim,), dtype=np.float32))
            states.append(spaces.Box(low=-5, high=3, shape=(self.agent_state_dim,), dtype=np.float32))

        self.action_space = spaces.Tuple(tuple(actions))
        self.observation_space = spaces.Tuple(tuple(observations))
        self.state_space = spaces.Tuple(tuple(states))

        # 初始化可视化
        self.visualizer = Visualize(self.args)

        # 初始化环境，基于roboEnv
        self.env = roboEnv(self, args)

        # 初始化通信网络 - 衡量哪些智能体是需要进行interaction
        self.adj_matrix = 1 - np.identity(self.num_robots, dtype=int)

    def _generate_step_goal_positions(self, actions):
        '''
        User implemented
        Calculates the goal locations for the current agent poses and actions
        returns an array of the robotarium positions that it is trying to reach
        '''
        goal = copy.deepcopy(self.agent_poses)
        for i, agent in enumerate(self.agents):
            goal[:, i] = agent.generate_goal(goal[:, i], actions[i], self.args)
        return goal

    def _update_tracking_and_locations(self, agent_actions):
        # iterate over all the prey
        for i, prey_location in enumerate(self.prey_loc):
            # if the prey has already been captured, nothing to be done
            if self.prey_captured[i]:
                continue
                # check if the prey has not been sensed
            if not self.prey_sensed[i]:
                # check if any of the agents has sensed it in the current step
                for agent in self.agents:
                    # check if any robot has it within its sensing radius
                    # print(self.agents.agent_poses[:2, agent.index], prey_location, np.linalg.norm(self.agents.agent_poses[:2, agent.index] - prey_location))
                    if np.linalg.norm(self.agent_poses[:2, agent.index] - prey_location) <= agent.sensing_radius:
                        self.prey_sensed[i] = True
                        break

            if self.prey_sensed[i]:
                # iterative over the agent_actions determined for each agent 
                for a, action in enumerate(agent_actions):
                    # check if any robot has no_action and has the prey within its capture radius if it is sensed already
                    if self.action_id2w[action] == 'no_action' \
                            and np.linalg.norm(self.agent_poses[:2, self.agents[a].index] - prey_location) <= \
                            self.agents[a].capture_radius:
                        self.prey_captured[i] = True
                        break

    def _generate_state_space(self):
        '''
        Generates a dictionary describing the state space of the robotarium
        x: Poses of all the robots
        '''
        state_space = {}
        state_space['poses'] = self.agent_poses
        state_space['num_prey'] = self.num_prey - sum(self.prey_captured)  # number of prey not captured
        state_space['unseen_prey'] = self.num_prey - sum(self.prey_sensed)  # number of prey unseen till now
        state_space['prey'] = []

        # return locations of all prey that are not captured  till now
        for i in range(self.num_prey):
            if not self.prey_captured[i]:
                state_space['prey'].append(np.array(self.prey_loc[i]).reshape((2, 1)))
        return state_space

    def load_agents(self):
        """
        Load the pre-defined agents / coalitions
        """
        t = "train"
        if self.args.test:
            t = "test"
        agents = []

        # sample a new coalition
        # 是否规定选取第几个coalition
        if self.args.manual_coalition_selection:
            coalition_idx = self.args.coalition_selection
        else:
            # 因为在coalition file里，专属于n个agent的scenario下，有n个coalition
            coalition_idx = np.random.randint(self.args.n_coalitions)

        s = str(self.num_robots) + "_agents"
        capture_agents = self.predefined_coalition[t]["coalitions"][s][coalition_idx]["capture"]
        predator_agents = self.predefined_coalition[t]["coalitions"][s][coalition_idx]["predator"]

        index = 0
        for idx, agent in capture_agents.items():
            agents.append(Agent(index, 0, agent["capture_radius"], self.action_id2w, self.action_w2id, self.args))
            index += 1
        for idx, agent in predator_agents.items():
            agents.append(Agent(index, agent["sensing_radius"], 0, self.action_id2w, self.action_w2id, self.args))
            index += 1
        return agents

    def reset(self):
        '''
        Resets the simulation
        '''
        # episode数量+1
        self.episode_number += 1
        # 是否在每个episode重新sample agent from the predefined coalitions
        if self.args.resample:
            self.agents = self.load_agents()

        self.episode_steps = 0
        self.episode_return = 0

        self.prey_locs = []
        self.num_prey = self.args.num_prey

        # 场地长宽
        width = self.args.ROBOT_INIT_RIGHT_THRESH - self.args.LEFT
        height = self.args.DOWN - self.args.UP
        # 根据robot数量，初始最小距离和场地长宽 生成agents的初始位置 --》3xN numpy array (of poses) [x,y,theta]
        self.agent_poses = generate_initial_locations(self.num_robots, width, height, self.args.ROBOT_INIT_RIGHT_THRESH,
                                                      start_dist=self.args.start_dist)

        # 根据robot数量，初始最小距离和场地长宽 生成preys的初始位置 --》3xN numpy array (of poses) [x,y,theta]
        width = self.args.RIGHT - self.args.PREY_INIT_LEFT_THRESH
        self.prey_loc = generate_initial_locations(self.num_prey, width, height, self.args.ROBOT_INIT_RIGHT_THRESH,
                                                   start_dist=self.args.step_dist, spawn_left=False)
        self.prey_loc = self.prey_loc[:2].T

        self.prey_captured = [False] * self.num_prey
        self.prey_sensed = [False] * self.num_prey

        self.state_space = self._generate_state_space()
        self.env.reset()

        return [[0] * self.agent_obs_dim] * self.num_robots

    def step(self, actions_):
        '''-+
        Step into the environment
        Returns observation, reward, done, info
        '''
        terminated = False
        if self.episode_steps == 0:
            assert self.episode_return == 0, "Episode return is not 0 at the start of the episode"
        self.episode_steps += 1

        # call the environment step function and get the updated state
        return_message = self.env.step(actions_)

        # 更新这一个step的prey的位置和状态
        self._update_tracking_and_locations(actions_)

        updated_state = self._generate_state_space()

        # get the observation and reward from the updated state
        obs = self.get_observations(updated_state)
        rewards = float(self.get_rewards(updated_state))

        # penalize for collisions, record in info
        violation_occurred = 0
        if self.args.penalize_violations:
            if self.args.end_ep_on_violation and return_message != '':
                violation_occurred += 1
                # print("violation: ", return_message)
                # rewards += self.args.violation_penalty
                rewards += -5.0
                terminated = True
            elif not self.args.end_ep_on_violation:
                violation_occurred = return_message
                rewards += np.log(
                    return_message + 1) * self.args.violation_penalty  # Taking the log because this can get out of control otherwise

        # terminate if needed
        if self.episode_steps >= self.args.max_episode_steps or \
                (updated_state['num_prey'] == 0 and self.args.terminate_on_success):
            terminated = True
        # 累计episodic reward
        self.episode_return += rewards

        info = {
            "terminated": terminated,
            "reward": rewards,
            "pct_captured_prey": sum(self.prey_captured) / self.num_prey,
            "total_prey": self.num_prey,
            "num_prey_captured": sum(self.prey_captured),
            "violation_occurred": violation_occurred,  # not a true count, just binary for if ANY violation occurred
        }
        if terminated:
            info["episode_return"] = self.episode_return
            info["episode_steps"] = self.episode_steps
            if return_message != '':
                info['reason'] = 'collision'
            elif self.episode_steps > self.args.max_episode_steps:
                info['reason'] = 'timelimit'
            elif updated_state['num_prey'] == 0:
                info['reason'] = 'capture all prey'
            pass
        # if not check_obs_dimensions(obs, self.num_robots, self.observation_space[0].shape[0]) or \
        #         not check_reward_dimensions([rewards] * self.num_robots, self.num_robots) or \
        #         not check_terminated_dimensions([terminated] * self.num_robots, self.num_robots) or \
        #         not check_info_dimensions([info] * self.num_robots, self.num_robots):
        #     print('?')
        assert check_obs_dimensions(obs, self.num_robots, self.observation_space[0].shape[0]), 'obs dim wrong'
        assert check_reward_dimensions([rewards] * self.num_robots, self.num_robots), 'reward dim wrong'
        assert check_terminated_dimensions([terminated] * self.num_robots, self.num_robots), 'done dim wrong'
        assert check_info_dimensions([info] * self.num_robots, self.num_robots), 'info dim wrong'

        return obs, [rewards] * self.num_robots, [terminated] * self.num_robots, [info] * self.num_robots

    def get_observations(self, state_space):
        '''
        Input: Takes in the current state space of the environment
        Outputs:
            an array with [agent_x_pos, agent_y_pos, sensed_prey_x_pose, sensed_prey_y_pose, sensing_radius, capture_radius]
            concatenated with the same array for the nearest neighbors based on args.delta or args.num_neighbors
        '''
        if self.prey_locs == []:
            for p in state_space['prey']:
                self.prey_locs = np.concatenate((self.prey_locs, p.reshape((1, 2))[0]))
        # iterate over all agents and store the observations for each in a dictionary
        # dictionary uses agent index as key
        observations = []
        neighbors = []  # Stores the neighbors of each agent if delta > -1
        for agent in self.agents:
            observations.append(agent.get_observation(state_space, self.agents))
            if self.args.delta > -1:
                neighbors.append(delta_disk_neighbors(state_space['poses'], agent.index, self.args.delta))

        # Updates the adjacency matrix
        if self.args.delta > -1:
            self.adj_matrix = np.zeros((self.num_robots, self.num_robots))
            for agents, ns in enumerate(neighbors):
                self.adj_matrix[agents, ns] = 1

        return observations

    def get_rewards(self, state_space):
        # Fully shared reward, this is a collaborative environment.
        reward = 0
        reward += (self.state_space['unseen_prey'] - state_space['unseen_prey']) * self.args.sense_reward
        reward += (self.state_space['num_prey'] - state_space['num_prey']) * self.args.capture_reward

        # if all the prey have been captured, don't penalize the agents anymore.
        if not self.args.terminate_on_success and state_space['num_prey'] == 0:
            reward += 0
        else:
            reward += self.args.time_penalty
        self.state_space = state_space
        return reward

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_state_space(self):
        return self.state_space
