from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from rlpd.types import PRNGKey


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions_vae(rng, apply_fn, params, observations: np.ndarray, z: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, z)
    if type(dist) is tuple:
        dist = dist[0]
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions_vae(apply_fn, params, observations: np.ndarray, z: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations, z)
    return dist.mode()


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations)
    if type(dist) is tuple:
        dist = dist[0]
    return dist.sample(seed=key), rng


@partial(jax.jit, static_argnames="apply_fn")
def _eval_actions(apply_fn, params, observations: np.ndarray) -> np.ndarray:
    dist = apply_fn({"params": params}, observations)
    return dist.mode()

@partial(jax.jit, static_argnames="apply_fn")
def _get_q_val(rng, apply_fn, params, observations: np.ndarray, actions: np.ndarray):
    key, rng = jax.random.split(rng)
    qs = apply_fn(
        {"params": params},
        observations,
        actions,
        True,
        rngs={"dropout": key},
    )
    q = qs.mean(axis=0)
    return q, rng


class Agent(struct.PyTreeNode):
    actor: TrainState
    rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray):
        actions, new_rng = _sample_actions(
                self.rng, self.actor.apply_fn, self.actor.params, observations
            )
        return np.asarray(actions), self.replace(rng=new_rng)

    def get_q_val(self, observations: np.ndarray, actions: np.ndarray):
        q, new_rng = _get_q_val(self.rng, self.critic_final_reward.apply_fn, self.critic_final_reward.params, observations, actions)
        return q, self.replace(rng=new_rng)
