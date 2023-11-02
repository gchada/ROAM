from typing import Optional

import gym
import numpy as np
from dm_control import composer
from dmcgym import DMCGYM
from gym.envs.registration import register
from gym.wrappers import FlattenObservation



def import_register(env_name):
    if "Real" in env_name:
        import rail_walker_gym.envs.register_real
    if "Mujoco" in env_name:
        import rail_walker_gym.envs.register_mujoco


class ClipActionToRange(gym.ActionWrapper):

    def __init__(self, env, min_action, max_action):
        super().__init__(env)

        min_action = np.asarray(min_action)
        max_action = np.asarray(max_action)

        min_action = min_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        max_action = max_action + np.zeros(env.action_space.shape,
                                           dtype=env.action_space.dtype)

        min_action = np.maximum(min_action, env.action_space.low)
        max_action = np.minimum(max_action, env.action_space.high)

        self.action_space = gym.spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class GoalObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def observation(self, obs):
        np.append(obs, self._env.task._goal_site.pos[0])
        np.append(obs, self._env.task._goal_site.pos[1])
        return obs


class PosObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def observation(self, obs):
        np.append(obs, self._env.task.x)
        np.append(obs, self._env.task.y)
        return obs


def make_env(task_name: str,
             clip_actions: bool = True,
             target_linear_velocity: float = 1.0,
             depth=True,
             upper_height=0.360,
             lower_height=0.03,
             hurdle="",
             resets=False,
             difficulty=1):

    kp = 60
    kd = kp * 0.1

    task = SimpleRun(kp=kp, kd=kd, depth=depth, target_linear_velocity=target_linear_velocity, upper_height=upper_height, lower_height=lower_height, hurdle=hurdle, resets=resets, difficulty=difficulty)

    env = composer.Environment(task, strip_singleton_obs_buffer_dim=True)

    env = DMCGYM(env)

    env = gym.wrappers.TimeLimit(env, 50000)

    env = gym.wrappers.ClipAction(env)  # Just for numerical stability.

    if clip_actions:
        ACTION_OFFSET = np.asarray([0.2, 0.4, 0.4] * 4) * 3.0
        INIT_QPOS = benchmark.domains.a1.legged_mujoco.robots.a1.A1._INIT_QPOS
        env = ClipActionToRange(env, INIT_QPOS - ACTION_OFFSET,
                                INIT_QPOS + ACTION_OFFSET)

    env = AddPreviousActions(env, action_history=1)
    env = FlattenObservation(env)
    env = GoalObservationWrapper(env)
    env = PosObservationWrapper(env)

    return env


make_env.metadata = DMCGYM.metadata


def make_mujoco_env(env_name: str,
                    control_frequency: int = 0,
                    clip_actions: bool = True,
                    action_filter_high_cut: Optional[float] = -1,
                    action_history: int = 1,
                    robot: str = "a1",
                    max_steps: int = 10000,
                    task: str = "locomotion",
                    upper_height: float = 0.360,
                    lower_height: float = 0.03,
                    hurdle="",
                    resets=False,
                    difficulty=1) -> gym.Env:
    if env_name == "manipulation":
        env = gym.make("LocoBot-v0")
    elif env_name == "a1":
        env = make_env(env_name, upper_height=upper_height, lower_height=lower_height, hurdle=hurdle, resets=resets, difficulty=difficulty)
    else:
        import_register(env_name)
        env = gym.make(env_name)
    return env
