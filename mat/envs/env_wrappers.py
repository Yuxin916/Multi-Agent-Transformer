"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe

import numpy as np

from mat.utils.util import tile_images


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


# class GuardSubprocVecEnv(ShareVecEnv):
#     def __init__(self, env_fns, spaces=None):
#         """
#         envs: list of gym environments to run in subprocesses
#         """
#         self.waiting = False
#         self.closed = False
#         nenvs = len(env_fns)
#         self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
#         self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
#                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
#         for p in self.ps:
#             p.daemon = False  # could cause zombie process
#             p.start()
#         for remote in self.work_remotes:
#             remote.close()
#
#         self.remotes[0].send(('get_spaces', None))
#         observation_space, share_observation_space, action_space = self.remotes[0].recv()
#         ShareVecEnv.__init__(self, len(env_fns), observation_space,
#                              share_observation_space, action_space)
#
#     def step_async(self, actions):
#
#         for remote, action in zip(self.remotes, actions):
#             remote.send(('step', action))
#         self.waiting = True
#
#     def step_wait(self):
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rews), np.stack(dones), infos
#
#     def reset(self):
#         for remote in self.remotes:
#             remote.send(('reset', None))
#         obs = [remote.recv() for remote in self.remotes]
#         return np.stack(obs)
#
#     def reset_task(self):
#         for remote in self.remotes:
#             remote.send(('reset_task', None))
#         return np.stack([remote.recv() for remote in self.remotes])
#
#     def close(self):
#         if self.closed:
#             return
#         if self.waiting:
#             for remote in self.remotes:
#                 remote.recv()
#         for remote in self.remotes:
#             remote.send(('close', None))
#         for p in self.ps:
#             p.join()
#         self.closed = True


class SubprocVecEnv(ShareVecEnv):
    """
    This is the wrapper for MPE envs
    N > 1
    """

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame)


def worker(remote, parent_remote, env_fn_wrapper):
    """
    This is the worker function used by SubprocVecEnv to simulate environment
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    """
    This is the wrapper for mujuco, google football, and starcraft2 envs
    N > 1
    """

    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_num_agents', None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, dones, infos, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def shareworker(remote, parent_remote, env_fn_wrapper):
    """
    This is the worker function used by ShareSubprocVecEnv to simulate environment
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_num_agents':
            remote.send((env.n_agents))
        elif cmd == 'get_spaces':
            remote.send(
                (env.observation_space, env.share_observation_space, env.action_space))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv_robotarium(ShareVecEnv):
    def __init__(self, env_fn, env_args):
        """
        envs: list of gym environments to run in subprocesses
        和ShareSubprocVecEnv的区别是
        1. 输入的是一个env_fn，而不是env_fn的list，还有list of env_args for all envs
        2. 用的是env_worker_robotarium，而不是shareworker
        3. 创建process的时候多了将env_args传入env_fn的步骤
            intuitively 相比于ShareSubprocVecEnv里env_fn是调用了N次StarCraft2Env，这里是gym register了一次但是args是N个
        4. step_wait里的result和StarCraft2Env顺序不一样
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_args)
        from functools import partial
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=env_worker_robotarium,
                           args=(work_remote, remote, CloudpickleWrapper(partial(env_fn, **env_arg))))
                   for (work_remote, remote, env_arg) in zip(self.work_remotes, self.remotes, env_args)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        # 发送get_env_info消息给每一个env，去看env_worker函数，得到env_info
        self.remotes[0].send(('get_num_agents', None))
        self.n_agents = self.remotes[0].recv()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv(
        )
        ShareVecEnv.__init__(self, len(env_args), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        share_obs, avail_actions, obs, adj_matrix, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), \
               infos, np.stack(avail_actions)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = zip(*results)
        return np.stack(obs), np.stack(share_obs), np.stack(available_actions)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def env_worker_robotarium(remote, parent_remote, env_fn_wrapper):
    """
    This is the worker function used by ShareSubprocVecEnv_robotarium to simulate environment
    """
    parent_remote.close()
    # Make environment
    # 进入这里的时候，env_fn是一个函数，调用它可以得到一个环境 -》在env文件夹的__init__.py中
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            # Take a step in the environment
            reward, terminated, env_info = env.step(data)
            # log info on change or termination
            # if last_env_info != env_info or all(terminated):
            if 'bool' in terminated.__class__.__name__:
                if terminated:
                    ob, s_ob, available_action = env.reset()
                else:
                    ob = env.get_obs()
                    s_ob = env.get_state()
                    available_action = env.get_avail_actions()
            else:
                if np.all(terminated):
                    ob, s_ob, available_action = env.reset()
                else:
                    ob = env.get_obs()
                    s_ob = env.get_state()
                    available_action = env.get_avail_actions()

            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            adj_matrix = env.get_adj_matrix()
            assert np.array([ob]).all() == np.array([obs]).all(), "ob != obs"
            assert np.array([s_ob]).all() == np.array([state]).all(), "s_ob != state"
            assert np.array([available_action]).all() == np.array(
                [available_actions]).all(), "available_action != avail_actions"

            remote.send((
                # Data for the next timestep needed to pick an action
                state,
                avail_actions,
                obs,
                adj_matrix,
                # Rest of the data for the current timestep
                reward,
                terminated,
                env_info,
            ))

        elif cmd == "reset":
            ob, s_ob, available_action = env.reset()
            obs = env.get_obs()
            state = env.get_state()
            available_actions = env.get_avail_actions()

            assert np.array([ob]).all() == np.array([obs]).all(), "ob != obs"
            assert np.array([s_ob]).all() == np.array([state]).all(), "s_ob != state"
            assert np.array([available_action]).all() == np.array(
                [available_actions]).all(), "available_action != avail_actions"

            remote.send((ob, s_ob, available_actions))

        elif cmd == "close":
            env.close()
            remote.close()
            break

        elif cmd == 'get_num_agents':
            remote.send(env.get_num_agents())
        elif cmd == "get_env_info":
            # 在env文件夹的__init__.py中_GymmaWrapper继承的MultiAgentEnv类中有get_env_info函数
            # 你可真难找
            remote.send(env.get_env_info())
        elif cmd == 'get_spaces':
            remote.send(
                (env.get_spaces()))
        else:
            raise NotImplementedError


# def choosesimpleworker(remote, parent_remote, env_fn_wrapper):
#     parent_remote.close()
#     env = env_fn_wrapper.x()
#     while True:
#         cmd, data = remote.recv()
#         if cmd == 'step':
#             ob, reward, done, info = env.step(data)
#             remote.send((ob, reward, done, info))
#         elif cmd == 'reset':
#             ob = env.reset(data)
#             remote.send((ob))
#         elif cmd == 'reset_task':
#             ob = env.reset_task()
#             remote.send(ob)
#         elif cmd == 'close':
#             env.close()
#             remote.close()
#             break
#         elif cmd == 'render':
#             if data == "rgb_array":
#                 fr = env.render(mode=data)
#                 remote.send(fr)
#             elif data == "human":
#                 env.render(mode=data)
#         elif cmd == 'get_spaces':
#             remote.send(
#                 (env.observation_space, env.share_observation_space, env.action_space))
#         else:
#             raise NotImplementedError


# class ChooseSimpleSubprocVecEnv(ShareVecEnv):
#     def __init__(self, env_fns, spaces=None):
#         """
#         envs: list of gym environments to run in subprocesses
#         """
#         self.waiting = False
#         self.closed = False
#         nenvs = len(env_fns)
#         self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
#         self.ps = [Process(target=choosesimpleworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
#                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
#         for p in self.ps:
#             p.daemon = True  # if the main process crashes, we should not cause things to hang
#             p.start()
#         for remote in self.work_remotes:
#             remote.close()
#         self.remotes[0].send(('get_spaces', None))
#         observation_space, share_observation_space, action_space = self.remotes[0].recv()
#         ShareVecEnv.__init__(self, len(env_fns), observation_space,
#                              share_observation_space, action_space)
#
#     def step_async(self, actions):
#         for remote, action in zip(self.remotes, actions):
#             remote.send(('step', action))
#         self.waiting = True
#
#     def step_wait(self):
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rews), np.stack(dones), infos
#
#     def reset(self, reset_choose):
#         for remote, choose in zip(self.remotes, reset_choose):
#             remote.send(('reset', choose))
#         obs = [remote.recv() for remote in self.remotes]
#         return np.stack(obs)
#
#     def render(self, mode="rgb_array"):
#         for remote in self.remotes:
#             remote.send(('render', mode))
#         if mode == "rgb_array":
#             frame = [remote.recv() for remote in self.remotes]
#             return np.stack(frame)
#
#     def reset_task(self):
#         for remote in self.remotes:
#             remote.send(('reset_task', None))
#         return np.stack([remote.recv() for remote in self.remotes])
#
#     def close(self):
#         if self.closed:
#             return
#         if self.waiting:
#             for remote in self.remotes:
#                 remote.recv()
#         for remote in self.remotes:
#             remote.send(('close', None))
#         for p in self.ps:
#             p.join()
#         self.closed = True


# def chooseworker(remote, parent_remote, env_fn_wrapper):
#     parent_remote.close()
#     env = env_fn_wrapper.x()
#     while True:
#         cmd, data = remote.recv()
#         if cmd == 'step':
#             ob, s_ob, reward, done, info, available_actions = env.step(data)
#             remote.send((ob, s_ob, reward, done, info, available_actions))
#         elif cmd == 'reset':
#             ob, s_ob, available_actions = env.reset(data)
#             remote.send((ob, s_ob, available_actions))
#         elif cmd == 'reset_task':
#             ob = env.reset_task()
#             remote.send(ob)
#         elif cmd == 'close':
#             env.close()
#             remote.close()
#             break
#         elif cmd == 'render':
#             remote.send(env.render(mode='rgb_array'))
#         elif cmd == 'get_spaces':
#             remote.send(
#                 (env.observation_space, env.share_observation_space, env.action_space))
#         else:
#             raise NotImplementedError


# class ChooseSubprocVecEnv(ShareVecEnv):
#     def __init__(self, env_fns, spaces=None):
#         """
#         envs: list of gym environments to run in subprocesses
#         """
#         self.waiting = False
#         self.closed = False
#         nenvs = len(env_fns)
#         self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
#         self.ps = [Process(target=chooseworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
#                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
#         for p in self.ps:
#             p.daemon = True  # if the main process crashes, we should not cause things to hang
#             p.start()
#         for remote in self.work_remotes:
#             remote.close()
#         self.remotes[0].send(('get_spaces', None))
#         observation_space, share_observation_space, action_space = self.remotes[0].recv(
#         )
#         ShareVecEnv.__init__(self, len(env_fns), observation_space,
#                              share_observation_space, action_space)
#
#     def step_async(self, actions):
#         for remote, action in zip(self.remotes, actions):
#             remote.send(('step', action))
#         self.waiting = True
#
#     def step_wait(self):
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         obs, share_obs, rews, dones, infos, available_actions = zip(*results)
#         return np.stack(obs), np.stack(share_obs), np.stack(rews), np.stack(dones), infos, np.stack(available_actions)
#
#     def reset(self, reset_choose):
#         for remote, choose in zip(self.remotes, reset_choose):
#             remote.send(('reset', choose))
#         results = [remote.recv() for remote in self.remotes]
#         obs, share_obs, available_actions = zip(*results)
#         return np.stack(obs), np.stack(share_obs), np.stack(available_actions)
#
#     def reset_task(self):
#         for remote in self.remotes:
#             remote.send(('reset_task', None))
#         return np.stack([remote.recv() for remote in self.remotes])
#
#     def close(self):
#         if self.closed:
#             return
#         if self.waiting:
#             for remote in self.remotes:
#                 remote.recv()
#         for remote in self.remotes:
#             remote.send(('close', None))
#         for p in self.ps:
#             p.join()
#         self.closed = True
#
#
# # def chooseguardworker(remote, parent_remote, env_fn_wrapper):
#     parent_remote.close()
#     env = env_fn_wrapper.x()
#     while True:
#         cmd, data = remote.recv()
#         if cmd == 'step':
#             ob, reward, done, info = env.step(data)
#             remote.send((ob, reward, done, info))
#         elif cmd == 'reset':
#             ob = env.reset(data)
#             remote.send((ob))
#         elif cmd == 'reset_task':
#             ob = env.reset_task()
#             remote.send(ob)
#         elif cmd == 'close':
#             env.close()
#             remote.close()
#             break
#         elif cmd == 'get_spaces':
#             remote.send(
#                 (env.observation_space, env.share_observation_space, env.action_space))
#         else:
#             raise NotImplementedError


# class ChooseGuardSubprocVecEnv(ShareVecEnv):
#     def __init__(self, env_fns, spaces=None):
#         """
#         envs: list of gym environments to run in subprocesses
#         """
#         self.waiting = False
#         self.closed = False
#         nenvs = len(env_fns)
#         self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
#         self.ps = [Process(target=chooseguardworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
#                    for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
#         for p in self.ps:
#             p.daemon = False  # if the main process crashes, we should not cause things to hang
#             p.start()
#         for remote in self.work_remotes:
#             remote.close()
#         self.remotes[0].send(('get_spaces', None))
#         observation_space, share_observation_space, action_space = self.remotes[0].recv(
#         )
#         ShareVecEnv.__init__(self, len(env_fns), observation_space,
#                              share_observation_space, action_space)
#
#     def step_async(self, actions):
#         for remote, action in zip(self.remotes, actions):
#             remote.send(('step', action))
#         self.waiting = True
#
#     def step_wait(self):
#         results = [remote.recv() for remote in self.remotes]
#         self.waiting = False
#         obs, rews, dones, infos = zip(*results)
#         return np.stack(obs), np.stack(rews), np.stack(dones), infos
#
#     def reset(self, reset_choose):
#         for remote, choose in zip(self.remotes, reset_choose):
#             remote.send(('reset', choose))
#         obs = [remote.recv() for remote in self.remotes]
#         return np.stack(obs)
#
#     def reset_task(self):
#         for remote in self.remotes:
#             remote.send(('reset_task', None))
#         return np.stack([remote.recv() for remote in self.remotes])
#
#     def close(self):
#         if self.closed:
#             return
#         if self.waiting:
#             for remote in self.remotes:
#                 remote.recv()
#         for remote in self.remotes:
#             remote.send(('close', None))
#         for p in self.ps:
#             p.join()
#         self.closed = True


"""
Single Parallel Env
"""


class DummyVecEnv(ShareVecEnv):
    """
    This is the wrapper for MPE envs
    N = 1
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError


class ShareDummyVecEnv(ShareVecEnv):
    """
    This is the wrapper for mujuco, google football, and starcraft2 envs
    N = 1
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.n_agents = env.n_agents
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos, available_actions = map(
            np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        return obs, share_obs, rews, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(np.array, zip(*results))
        return obs, share_obs, available_actions

    def close(self):
        for env in self.envs:
            env.close()

    def save_replay(self):
        for env in self.envs:
            env.save_replay()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

# class ChooseDummyVecEnv(ShareVecEnv):
#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         ShareVecEnv.__init__(self, len(
#             env_fns), env.observation_space, env.share_observation_space, env.action_space)
#         self.actions = None
#
#     def step_async(self, actions):
#         self.actions = actions
#
#     def step_wait(self):
#         results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
#         obs, share_obs, rews, dones, infos, available_actions = map(
#             np.array, zip(*results))
#         self.actions = None
#         return obs, share_obs, rews, dones, infos, available_actions
#
#     def reset(self, reset_choose):
#         results = [env.reset(choose)
#                    for (env, choose) in zip(self.envs, reset_choose)]
#         obs, share_obs, available_actions = map(np.array, zip(*results))
#         return obs, share_obs, available_actions
#
#     def close(self):
#         for env in self.envs:
#             env.close()
#
#     def render(self, mode="human"):
#         if mode == "rgb_array":
#             return np.array([env.render(mode=mode) for env in self.envs])
#         elif mode == "human":
#             for env in self.envs:
#                 env.render(mode=mode)
#         else:
#             raise NotImplementedError

# class ChooseSimpleDummyVecEnv(ShareVecEnv):
#     def __init__(self, env_fns):
#         self.envs = [fn() for fn in env_fns]
#         env = self.envs[0]
#         ShareVecEnv.__init__(self, len(
#             env_fns), env.observation_space, env.share_observation_space, env.action_space)
#         self.actions = None
#
#     def step_async(self, actions):
#         self.actions = actions
#
#     def step_wait(self):
#         results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
#         obs, rews, dones, infos = map(np.array, zip(*results))
#         self.actions = None
#         return obs, rews, dones, infos
#
#     def reset(self, reset_choose):
#         obs = [env.reset(choose)
#                    for (env, choose) in zip(self.envs, reset_choose)]
#         return np.array(obs)
#
#     def close(self):
#         for env in self.envs:
#             env.close()
#
#     def render(self, mode="human"):
#         if mode == "rgb_array":
#             return np.array([env.render(mode=mode) for env in self.envs])
#         elif mode == "human":
#             for env in self.envs:
#                 env.render(mode=mode)
#         else:
#             raise NotImplementedError
