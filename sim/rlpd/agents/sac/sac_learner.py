"""Implementations of algorithms for continuous control."""

from functools import partial
from typing import Dict, Optional, Sequence, Tuple

import flax
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState

from rlpd.agents.agent import Agent
from rlpd.agents.sac.temperature import Temperature
from rlpd.data.dataset import DatasetDict
from rlpd.distributions import TanhNormal
from rlpd.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


class SACLearner(Agent):
    critic: TrainState
    critic_final_reward: TrainState
    target_critic_final_reward: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: Optional[int] = struct.field(
        pytree_node=False
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: Optional[int] = None,
        critic_dropout_rate: Optional[float] = None,
        critic_weight_decay: Optional[float] = None,
        critic_layer_norm: bool = False,
        target_entropy: Optional[float] = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
    ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_base_cls = partial(
            MLP, hidden_dims=hidden_dims, activate_final=True, use_pnorm=use_pnorm
        )
        actor_def = TanhNormal(actor_base_cls, action_dim)
        actor_params = actor_def.init(actor_key, observations)["params"]
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=actor_lr),
        )

        if use_critic_resnet:
            critic_base_cls = partial(
                MLPResNetV2,
                num_blocks=1,
            )
        else:
            critic_base_cls = partial(
                MLP,
                hidden_dims=hidden_dims,
                activate_final=True,
                dropout_rate=critic_dropout_rate,
                use_layer_norm=critic_layer_norm,
                use_pnorm=use_pnorm,
            )
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        if critic_weight_decay is not None:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
        else:
            tx = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        critic_final_reward = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=tx,
        )

        target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
        target_critic = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        target_critic_final_reward = TrainState.create(
            apply_fn=target_critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )

        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)["params"]
        temp = TrainState.create(
            apply_fn=temp_def.apply,
            params=temp_params,
            tx=optax.adam(learning_rate=temp_lr),
        )

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            critic_final_reward=critic_final_reward,
            target_critic_final_reward=target_critic_final_reward,
        )

    def update_actor_online(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        key, rng = jax.random.split(self.rng)
        key2, rng = jax.random.split(rng)

        def actor_loss_fn(actor_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic_final_reward.apply_fn(
                {"params": self.critic_final_reward.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)
            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q
            ).mean()
            return actor_loss, {"actor_loss": actor_loss, "entropy": -log_probs.mean()}

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> Tuple[Agent, Dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)
        critic = self.critic.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def update_critic_final_reward(self, batch: DatasetDict) -> Tuple[TrainState, Dict[str, float]]:

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic_final_reward.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic_final_reward.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = self.critic_final_reward.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic_final_reward.params)
        critic = self.critic_final_reward.apply_gradients(grads=grads)

        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic_final_reward.params, self.tau
        )
        target_critic = self.target_critic_final_reward.replace(params=target_critic_params)

        return self.replace(critic_final_reward=critic, target_critic_final_reward=target_critic, rng=rng), info


    def update_critic_final_reward_offline(self, batch: DatasetDict, labels: jnp.ndarray, agents, reg_weight) -> Tuple[TrainState, Dict[str, float]]:

        rng = self.rng

        dist = self.actor.apply_fn(
            {"params": self.actor.params}, batch["next_observations"]
        )

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key, self.target_critic_final_reward.params, self.num_min_qs, self.num_qs
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic_final_reward.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        def critic_loss_fn_friction(critic_params, critic_params_0, critic_params_1, critic_params_2, critic_params_3):
            critic_params_arr = [critic_params_0, critic_params_1, critic_params_2, critic_params_3]

            rng = self.rng
            key, rng = jax.random.split(rng)
            qs = self.critic_final_reward.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True

            vs = []
            for pos in range(len(agents)):
                agent = agents[pos]
                agent_qs = []
                for action_num in range(5):
                    key, rng = jax.random.split(rng)
                    dist = agent.actor.apply_fn({"params": agent.actor.params}, batch["observations"])
                    if type(dist) is tuple:
                        dist = dist[0]
                    sampled_actions = dist.sample(seed=key)
                    key, rng = jax.random.split(rng)
                    q_val = agent.critic_final_reward.apply_fn(
                        {"params": critic_params_arr[pos]},
                        batch["observations"],
                        sampled_actions,
                        True,
                        rngs={"dropout": key},
                    ).mean(axis=0)
                    agent_qs.append(q_val)
                v = jnp.mean(jnp.asarray(agent_qs), axis=0)
                vs.append(v)
            values = jnp.asarray(vs)
            values = jnp.transpose(values)
            ce_loss = optax.softmax_cross_entropy(values, labels).mean()
            bellman_loss = ((qs - target_q) ** 2).mean()
            critic_loss = bellman_loss + reg_weight * ce_loss
            return critic_loss, {"ce_loss": ce_loss, "bellman_loss": bellman_loss}

        def critic_loss_fn_stiffness(critic_params, critic_params_0, critic_params_1, critic_params_2, critic_params_3, critic_params_4, critic_params_5, critic_params_6, critic_params_7, critic_params_8):
            critic_params_arr = [critic_params_0, critic_params_1, critic_params_2, critic_params_3, critic_params_4, critic_params_5, critic_params_6, critic_params_7, critic_params_8]

            rng = self.rng
            key, rng = jax.random.split(rng)
            qs = self.critic_final_reward.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True

            vs = []
            for pos in range(len(agents)):
                agent = agents[pos]
                agent_qs = []
                for action_num in range(5):
                    key, rng = jax.random.split(rng)
                    dist = agent.actor.apply_fn({"params": agent.actor.params}, batch["observations"])
                    if type(dist) is tuple:
                        dist = dist[0]
                    sampled_actions = dist.sample(seed=key)
                    key, rng = jax.random.split(rng)
                    q_val = agent.critic_final_reward.apply_fn(
                        {"params": critic_params_arr[pos]},
                        batch["observations"],
                        sampled_actions,
                        True,
                        rngs={"dropout": key},
                    ).mean(axis=0)
                    agent_qs.append(q_val)
                v = jnp.mean(jnp.asarray(agent_qs), axis=0)
                vs.append(v)
            values = jnp.asarray(vs)
            values = jnp.transpose(values)
            ce_loss = optax.softmax_cross_entropy(values, labels).mean()
            bellman_loss = ((qs - target_q) ** 2).mean()
            critic_loss = bellman_loss + reg_weight * ce_loss
            return critic_loss, {"ce_loss": ce_loss, "bellman_loss": bellman_loss}

        critic_params_arr = []
        for agent in agents:
            critic_params_arr.append(agent.critic_final_reward.params)
        argnums = list(range(len(agents) + 1))
        grads, info = jax.grad(critic_loss_fn_stiffness if len(agents) == 9 else critic_loss_fn_friction, has_aux=True, argnums=argnums)(self.critic_final_reward.params, *critic_params_arr)

        return grads, info


    @partial(jax.jit, static_argnames="utd_ratio")
    def update(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)
            new_agent, critic_final_reward_info = new_agent.update_critic_final_reward(mini_batch)

        new_agent, actor_info = new_agent.update_actor_online(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **critic_final_reward_info, **temp_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update_offline(self, batch: DatasetDict, utd_ratio: int, agents, labels, reg_weight):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)
            mini_labels = jax.tree_util.tree_map(slice, labels)

            grads, critic_final_reward_info = new_agent.update_critic_final_reward_offline(mini_batch, mini_labels, agents, reg_weight)

        return grads, {**critic_final_reward_info}


    @partial(jax.jit, static_argnames="utd_ratio")
    def update_actor(self, batch: DatasetDict, utd_ratio: int):

        new_agent = self
        for i in range(utd_ratio):

            def slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(slice, batch)

        new_agent, actor_info = new_agent.update_actor_online(mini_batch)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **temp_info}
