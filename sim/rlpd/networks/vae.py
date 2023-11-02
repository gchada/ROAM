from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax import random
import jax

default_init = nn.initializers.xavier_uniform


class VAE(nn.Module):
    beta: Optional[float] = 0.5
    state_only: Optional[bool] = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, actions: jnp.ndarray = None) -> jnp.ndarray:
        if not self.state_only:
            x = jnp.concatenate([x, actions], axis=-1)
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(500, name='fc2')(x)
        x = nn.relu(x)
        mean = nn.Dense(24, name='fc3_mean')(x)
        logvar = nn.Dense(24, name='fc3_logvar')(x)
        std = jnp.exp(0.5 * logvar)
        rng = random.PRNGKey(0)
        eps = random.normal(rng, logvar.shape)
        z = mean + eps * std
        kl = -self.beta * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
        return z, kl, mean, std
