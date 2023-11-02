import collections
from typing import Optional, Union

import gym
import gym.spaces
import jax
import numpy as np

from jaxrl5.data.dataset import Dataset, DatasetDict


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys()
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=np.float32),
            dones=np.empty((capacity,), dtype=bool),
            labels=np.empty((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self._insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.dataset_dict['observations'])

        eps = 1e-6
        lim = 1 - eps
        dataset.dataset_dict["actions"] = np.clip(dataset.dataset_dict["actions"], -lim, lim)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self._capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        for k in self.dataset_dict.keys():
            self.dataset_dict[k][:num_samples] = dataset.dataset_dict[k][:num_samples]


        
        self._insert_index = num_samples
        self._size = num_samples

    def get_iterator(self, queue_size: int = 2, sample_args: dict = {}):
        # See https://flax.readthedocs.io/en/latest/_modules/flax/jax_utils.html#prefetch_to_device
        # queue_size = 2 should be ok for one GPU.

        queue = collections.deque()

        def enqueue(n):
            for _ in range(n):
                data = self.sample(**sample_args)
                queue.append(jax.device_put(data))

        enqueue(queue_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

    def add(self, buffer, index=0, buffer_source=None):
        observations = buffer.dataset_dict['observations']
        next_observations = buffer.dataset_dict['next_observations']
        actions = buffer.dataset_dict['actions']
        rewards = buffer.dataset_dict['rewards']
        masks = buffer.dataset_dict['masks']
        dones = buffer.dataset_dict['dones']
        labels = np.full_like(masks, buffer_source)

        for i in range(index, buffer._size):
            self.insert(
                dict(observations=observations[i],
                     actions=actions[i],
                     rewards=rewards[i],
                     masks=masks[i],
                     dones=dones[i],
                     next_observations=next_observations[i],
                     labels=labels[i]))

    def add_num(self, buffer, index=0, num_transitions=0, buffer_source=None):
        observations = buffer.dataset_dict['observations']
        next_observations = buffer.dataset_dict['next_observations']
        actions = buffer.dataset_dict['actions']
        rewards = buffer.dataset_dict['rewards']
        masks = buffer.dataset_dict['masks']
        dones = buffer.dataset_dict['dones']
        labels = np.full_like(masks, buffer_source)

        for i in range(index, index + num_transitions):
            self.insert(
                dict(observations=observations[i],
                        actions=actions[i],
                        rewards=rewards[i],
                        masks=masks[i],
                        dones=dones[i],
                        next_observations=next_observations[i],
                        labels=labels[i]))