from typing import Dict

import gym
import numpy as np

from jaxrl5.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)

    for i in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, _ = env.step(action)
    return {"return": np.mean(env.return_queue), "std": np.std(env.return_queue), "length": np.mean(env.length_queue)}


def evaluate_validation_error(
    agent, replay_buffer
) -> Dict[str, float]:
    validation_size = len(replay_buffer)
    running_size = 0
    running_td_error = 0

    batch_size = 512
    num_batches = validation_size // batch_size

    for i in range(num_batches):
        batch = replay_buffer.sample(
            batch_size,
            indx=np.arange(i * batch_size, (i + 1) * batch_size)
        )
        td_error = agent.compute_td_error(batch)
        running_td_error += np.sum(td_error)
        running_size += batch_size

    # Handle the remaining samples
    remaining_samples = validation_size - num_batches * batch_size
    if remaining_samples > 0:
        batch = replay_buffer.sample(
            remaining_samples,
            indx=np.arange(num_batches * batch_size, validation_size)
        )
        td_error = agent.compute_td_error(batch)
        running_td_error += np.sum(td_error)
        running_size += remaining_samples

    return {"validation_td_error": running_td_error / running_size}