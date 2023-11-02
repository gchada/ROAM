from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.training.train_state import TrainState

from jaxrl5.types import PRNGKey


@partial(jax.jit, static_argnames="apply_fn")
def _sample_actions(rng, apply_fn, params, observations: np.ndarray, **kwargs) -> np.ndarray:
    key, rng = jax.random.split(rng)
    dist = apply_fn({"params": params}, observations, **kwargs)
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

    def eval_actions(self, observations: np.ndarray, **kwargs) -> np.ndarray:
        actions = _eval_actions(self.actor.apply_fn, self.actor.params, observations, **kwargs)
        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray, **kwargs) -> np.ndarray:
        actions, new_rng = _sample_actions(
            self.rng, self.actor.apply_fn, self.actor.params, observations, **kwargs
        )
        return np.asarray(actions), self.replace(rng=new_rng)
    
    def get_q_val(self, observations: np.ndarray, actions: np.ndarray):
        q, new_rng = _get_q_val(self.rng, self.critic.apply_fn, self.critic.params, observations, actions)
        return q, self.replace(rng=new_rng)
    