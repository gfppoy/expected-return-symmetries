""" 
Based on PureJaxRL Implementation of PPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.traverse_util as traverse_util
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, OvercookedV2LogWrapper
from jaxmarl.environments import overcooked_v2_layouts
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import os
import wandb
import functools
import math
import pickle

def save_permutation(in_permutation, out_permutation, config, step=1):
    os.makedirs(config["SAVE_CHECKPOINT"], exist_ok=True)
    in_permutation_layer_1_kernel_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_1_kernel_checkpoint_{step}.pkl")
    in_permutation_layer_1_bias_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_1_bias_checkpoint_{step}.pkl")
    in_permutation_layer_2_kernel_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_2_kernel_checkpoint_{step}.pkl")
    in_permutation_layer_2_bias_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_2_bias_checkpoint_{step}.pkl")
    in_permutation_layer_3_kernel_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_3_kernel_checkpoint_{step}.pkl")
    in_permutation_layer_3_bias_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"in_permutation_layer_3_bias_checkpoint_{step}.pkl")
    out_permutation_checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"out_permutation_checkpoint_{step}.pkl")

    with open(in_permutation_layer_1_kernel_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[0]['kernel'], f)

    with open(in_permutation_layer_1_bias_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[0]['bias'], f)

    with open(in_permutation_layer_2_kernel_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[1]['kernel'], f)

    with open(in_permutation_layer_2_bias_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[1]['bias'], f)

    with open(in_permutation_layer_3_kernel_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[2]['kernel'], f)

    with open(in_permutation_layer_3_bias_checkpoint_file, 'wb') as f:
        pickle.dump(in_permutation[2]['bias'], f)

    with open(out_permutation_checkpoint_file, 'wb') as f:
        pickle.dump(out_permutation, f)

def identity_init():
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Shape must be square for a permutation matrix.")
        return jnp.eye(shape[0], dtype=dtype)
    return init

def approximate_identity_init(scale=1e-2):
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Shape must be square for a permutation matrix.")
        identity = jnp.eye(shape[0], dtype=dtype)
        key, _key = jax.random.split(key)
        noise = scale * jax.random.normal(_key, shape, dtype=dtype)  # Gaussian noise
        return identity + noise
    return init

# Example for permutation layers
def permute_two_indices_init(index1, index2):
    def init(key, shape, dtype=jnp.float32):  # Explicitly set dtype
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] != 6:
            raise ValueError("Shape must be square for a permutation matrix.")
        perm_mat = jnp.eye(6, dtype=dtype)
        row1 = perm_mat[index1, :].copy()
        row2 = perm_mat[index2, :].copy()
        perm_mat = perm_mat.at[index1, :].set(row2)
        perm_mat = perm_mat.at[index2, :].set(row1)
        return perm_mat

    return init

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    return params_dict

def update_params(init_params, loaded_params):
    # Flatten both parameter trees
    flat_init_params = traverse_util.flatten_dict(init_params)
    flat_loaded_params = traverse_util.flatten_dict(loaded_params)

    # Mapping from new layer names to old ones
    mapping = {
        'Dense_3': 'Dense_0',
        'Dense_4': 'Dense_1',
        'Dense_6': 'Dense_2',
        'Dense_7': 'Dense_3'
    }

    # Combine parameters, including old CNN and ScannedRNN, but keeping new Dense layers initialized
    combined_params = {}
    for key in flat_init_params.keys():
        # Map the new dense layers to the old layers
        mapped_key = mapping.get(key[1], key)
        if isinstance(mapped_key, str):
            mapped_key = (key[0], mapped_key, key[2])
        
        # Exclude the new Dense layers from copying the old params
        if key[1] not in {'Dense_0', 'Dense_1', 'Dense_2', 'Dense_5'}:
            if mapped_key in flat_loaded_params:
                # Use old parameters for renamed dense layers, CNN, and ScannedRNN
                combined_params[key] = flat_loaded_params[mapped_key]
            else:
                # If no mapping, use the new parameters
                combined_params[key] = flat_init_params[key]
        else:
            # For new Dense layers (Dense_0, Dense_1, Dense_2), keep freshly initialized params
            combined_params[key] = flat_init_params[key]

    # Unflatten to get the final parameters structure
    return traverse_util.unflatten_dict(combined_params)

def selective_zero_grads(grads, freeze_layers=('Dense_0', 'Dense_1', 'Dense_2', 'Dense_6', 'Dense_7')):
    """
    Zero out gradients for all parameters except those in specified layers.
    """
    # Correct the freeze_layers tuple if necessary
    freeze_layers = tuple(freeze_layers)

    def mask_grad(path, grad):
        if path[0] != 'params':
            return grad  # Do not modify non-parameter gradients
        layer_name = path[1]
        if layer_name in ('CNN_0', 'ScannedRNN_0'):
            return jnp.zeros_like(grad)  # Zero out gradients in CNN and RNN layers
        elif layer_name in freeze_layers:
            return grad  # Keep gradients for specified layers
        else:
            return jnp.zeros_like(grad)  # Zero out other gradients

    flat_grads = traverse_util.flatten_dict(grads)
    masked_flat_grads = {path: mask_grad(path, grad) for path, grad in flat_grads.items()}
    processed_grads = traverse_util.unflatten_dict(masked_flat_grads)
    return processed_grads


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size, dtype=jnp.float32)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            dtype=jnp.float32,
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x

class ActorCriticRNNNoSymm(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    temperature: float = 1.1

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = obs

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean / self.temperature)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    temperature: float = 1.1

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        n = 5 * 5 * 38

        batch_dims = obs.shape[:-3]
        flattened_obs = jnp.reshape(obs, (*batch_dims, -1)).astype(jnp.float32)

        # in_permutation_layer_1 = nn.Dense(n, kernel_init=approximate_identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        in_permutation_layer_1 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        hidden_obs1 = nn.relu(in_permutation_layer_1(flattened_obs))

        # in_permutation_layer_2 = nn.Dense(n, kernel_init=approximate_identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        in_permutation_layer_2 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        hidden_obs2 = nn.relu(in_permutation_layer_2(hidden_obs1))

        # in_permutation_layer_3 = nn.Dense(n, kernel_init=approximate_identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        in_permutation_layer_3 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0), dtype=jnp.float32)
        transformed_obs = nn.relu(in_permutation_layer_3(hidden_obs2))

        embedding = jnp.reshape(transformed_obs, obs.shape)

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # out_permutation = nn.Dense(self.action_dim, kernel_init=approximate_identity_init(), use_bias=False, dtype=jnp.float32)
        out_permutation = nn.Dense(self.action_dim, kernel_init=identity_init(), use_bias=False, dtype=jnp.float32)
        # out_permutation = nn.Dense(self.action_dim, kernel_init=permute_two_indices_init(index1=4, index2=5), use_bias=False, dtype=jnp.float32)
        transformed_actor_mean = out_permutation(actor_mean)

        pi = distrax.Categorical(logits=transformed_actor_mean / self.temperature)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation)(x)

        actor_mean = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = OvercookedV2LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def train(rng, params):

        # INIT NETWORK
        network_no_symm = ActorCriticRNNNoSymm(env.action_space(env.agents[0]).n, config=config)
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, config["NUM_ENVS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["GRU_HIDDEN_DIM"]
        )

        net_p = network.init(_rng, init_hstate, init_x)

        network_params = []
        network_params_no_symm = []
        for i, p in enumerate(params):
            net_p = network.init(_rng, init_hstate, init_x)
            network_params_no_symm.append(p)
            updated_params = update_params(net_p, p)
            if i == 0:
                in_permutation_layer_1 = updated_params['params']['Dense_0']
                in_permutation_layer_2 = updated_params['params']['Dense_1']
                in_permutation_layer_3 = updated_params['params']['Dense_2']
                out_permutation = updated_params['params']['Dense_5']['kernel']
            else:
                updated_params['params']['Dense_0'] = in_permutation_layer_1
                updated_params['params']['Dense_1'] = in_permutation_layer_2
                updated_params['params']['Dense_2'] = in_permutation_layer_3
                updated_params['params']['Dense_5']['kernel'] = out_permutation

            network_params.append(updated_params)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(create_learning_rate_fn(), eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_states = []
        train_states_no_symm = []
        for net_p in network_params:
            train_states.append(
                TrainState.create(
                    apply_fn=network.apply,
                    params=net_p,
                    tx=tx,
                )
            )
        for net_p in network_params_no_symm:
            train_states_no_symm.append(
                TrainState.create(
                    apply_fn=network_no_symm.apply,
                    params=net_p,
                    tx=tx,
                )
            )

        train_states = [train_states_no_symm[0], train_states[1]]

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        init_hstates = []
        for _ in range(2):
            init_hstate = ScannedRNN.initialize_carry(
                config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
            )
            init_hstates.append(init_hstate)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstate,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                hstate1, pi1, value1 = network_no_symm.apply(train_state[0].params, hstate[0], ac_in)
                hstate2, pi2, value2 = network.apply(train_state[1].params, hstate[1], ac_in)

                action1 = pi1.sample(seed=_rng)
                action2 = pi2.sample(seed=_rng)

                log_prob1 = pi1.log_prob(action1)
                log_prob2 = pi2.log_prob(action2)

                env_act_a = unbatchify(action1, env.agents, config["NUM_ENVS"], env.num_agents)
                env_act_b = unbatchify(action2, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = {"agent_0": env_act_a["agent_0"], "agent_1": env_act_b["agent_1"]}

                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor, reward, info["shaped_reward"]
                )

                shaped_reward = jnp.array(
                    [info["shaped_reward"][a] for a in env.agents]
                )
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()

                transition1 = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action1.squeeze(),
                    value1.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob1.squeeze(),
                    obs_batch,
                    info,
                )
                transition2 = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action2.squeeze(),
                    value2.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob2.squeeze(),
                    obs_batch,
                    info,
                )
                
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    [hstate1, hstate2],
                    rng,
                )
                return runner_state, (transition1, transition2)


            initial_hstate1, initial_hstate2 = runner_state[-2]
            runner_state, (traj_batch1, traj_batch2) = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, update_step, hstate, rng = (
                runner_state
            )

            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            ac_in = (
                last_obs_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )

            _, _, last_val1 = network_no_symm.apply(train_state[0].params, hstate[0], ac_in)
            _, _, last_val2 = network.apply(train_state[1].params, hstate[1], ac_in)

            last_val1 = last_val1.squeeze()
            last_val2 = last_val2.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages1, targets1 = _calculate_gae(traj_batch1, last_val1)
            advantages2, targets2 = _calculate_gae(traj_batch2, last_val2)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )

                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    processed_grads = selective_zero_grads(grads)
                    train_state = train_state.apply_gradients(grads=processed_grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = (
                    update_state
                )
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (1, config["NUM_ACTORS"], -1))
                batch = (
                    init_hstate,
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate.squeeze(),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            update_state = (
                train_state[1],
                initial_hstate2,
                traj_batch2,
                advantages2,
                targets2,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            new_train_state_1 = update_state[0]
            new_train_state = [train_state[0], new_train_state_1]

            metric = traj_batch1.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            jax.debug.callback(callback, metric)

            runner_state = (
                new_train_state,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstate,
                rng,
            )

            return runner_state, metric

        runner_state = (
            train_states,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            init_hstates,
            rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


@hydra.main(
    version_base=None, config_path="config", config_name="xp_ippo_rnn_overcooked_v2_neural_symm_disc"
)
def main(config):
    config = OmegaConf.to_container(config)

    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", "OvercookedV2"],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"ippo_rnn_overcooked_v2_{layout_name}",
    )

    params = []
    for checkpoint in config["LOAD_CHECKPOINT"].split(','):
        params.append(load_checkpoint(checkpoint))

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config))
        out = jax.vmap(train_jit)(rngs, params) # for symmetry learning

    save_permutation((jax.device_get(out["runner_state"][0][1]).params['params']['Dense_0'], jax.device_get(out["runner_state"][0][1]).params['params']['Dense_1'], jax.device_get(out["runner_state"][0][1]).params['params']['Dense_2']), 
                        jax.device_get(out["runner_state"][0][1]).params['params']['Dense_5']['kernel'], 
                        config)

if __name__ == "__main__":
    main()
