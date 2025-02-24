"""
Based on PureJaxRL Implementation of PPO
"""

import os
import pickle

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.traverse_util as traverse_util
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL as HanabiGame
import jaxmarl
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

def permute_two_indices_init(index1, index2):
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1] or shape[0] != 21:
            raise ValueError("Shape must be square for a permutation matrix, and act on 21 dimensional action space.")
        perm_mat = jnp.eye(21, dtype=dtype)
        
        row1 = perm_mat[index1, :].copy()
        row2 = perm_mat[index2, :].copy()
        
        perm_mat = perm_mat.at[index1, :].set(row2)
        perm_mat = perm_mat.at[index2, :].set(row1)
        return perm_mat

    return init

def identity_init():
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Shape must be square for a permutation matrix.")
        return jnp.eye(shape[0], dtype=dtype)
    return init


def save_permutation(in_permutation, out_permutation, config, step):
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

def save_checkpoint(params, config, step):
    os.makedirs(config["SAVE_CHECKPOINT"], exist_ok=True)
    checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"checkpoint_{step}.pkl")

    params_dict = jax.tree_util.tree_map(lambda x: np.array(x), params)

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(params_dict, f)

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    return params_dict

def update_params(init_params, loaded_params):
    # Flatten both parameter trees
    flat_init_params = traverse_util.flatten_dict(init_params)
    flat_loaded_params = traverse_util.flatten_dict(loaded_params)

    mapping = {
        'Dense_3': 'Dense_0',
        'Dense_4': 'Dense_1',
        'Dense_5': 'Dense_2',
        'Dense_7': 'Dense_3',
        'Dense_8': 'Dense_4'
    }

    # Combine parameters, excluding the new layers
    combined_params = {}
    for key in flat_init_params.keys():
        mapped_key = mapping.get(key[1], key)
        if isinstance(mapped_key, str):
            mapped_key = (key[0], mapped_key, key[2])
        if mapped_key in flat_loaded_params and key[1] != 'Dense_0' and key[1] != 'Dense_1' and key[1] != 'Dense_2' and key[1] != 'Dense_6':
            combined_params[key] = flat_loaded_params[mapped_key]
        else:
            combined_params[key] = flat_init_params[key]

    # Unflatten to get the final parameters structure
    return traverse_util.unflatten_dict(combined_params)


def selective_zero_grads(grads, freeze_layers=('Dense_0', 'Dense_1', 'Dense_2', 'Dense_7', 'Dense_8')):
    """
    JIT-compatible function to zero out gradients for all parameters except those in specified layers.
    Args:
        grads (dict): A nested dictionary of gradients.
        freeze_layers (tuple of str): Layer names whose gradients should not be zeroed.

    Returns:
        dict: A nested dictionary of gradients with specified layers' gradients untouched and others zeroed.
    """
    def zero_out_grads(layer_name, grad):
        """
        Zero out gradients based on layer name. Uses JAX operations for JIT compatibility.
        """
        # Check if layer should be frozen (1 for update, 0 for freeze).
        freeze = jnp.array([layer_name in freeze_layers], dtype=jnp.float32)
        # Use JAX operations to zero out or keep the gradient.
        return jax.lax.cond(freeze[0], lambda _: grad, lambda _: jnp.zeros_like(grad), None)

    def process_layer_grads(layer_name, layer_grads):
        """
        Process a single layer's gradients, applying zero_out_grads to each parameter.
        """
        return jax.tree_map(lambda grad: zero_out_grads(layer_name, grad), layer_grads)

    # Process each layer's gradients, applying conditional zeroing.
    processed_grads = {layer: process_layer_grads(layer, layer_grads) for layer, layer_grads in grads['params'].items()}

    return {'params': processed_grads}

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions, num_compositions = x
        n = 658

        in_permutation_layer_1 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0))
        hidden_obs1 = nn.relu(in_permutation_layer_1(obs))

        in_permutation_layer_2 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0))
        hidden_obs2 = nn.relu(in_permutation_layer_2(hidden_obs1))

        in_permutation_layer_3 = nn.Dense(n, kernel_init=identity_init(), bias_init=constant(0.0))
        transformed_obs = nn.relu(in_permutation_layer_3(hidden_obs2))

        def apply_in_permutation(x):
            x = nn.relu(in_permutation_layer_1(x))
            x = nn.relu(in_permutation_layer_2(x))
            x = nn.relu(in_permutation_layer_3(x))
            return x
        transformed_obs = jax.lax.cond(num_compositions == 2, apply_in_permutation, lambda x: x, transformed_obs)

        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(transformed_obs)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        out_permutation = nn.Dense(self.action_dim, kernel_init=permute_two_indices_init(index1=0, index2=2), use_bias=False)
        transformed_actor_mean = out_permutation(actor_mean)
        def apply_out_permutation(x):
            return out_permutation(x)
        transformed_actor_mean = jax.lax.cond(num_compositions == 2, apply_out_permutation, lambda x: x, transformed_actor_mean)

        unavail_actions = 1 - avail_actions
        action_logits = transformed_actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
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
    avail_actions: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    env = HanabiGame()
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] // config["NUM_CHECKPOINTS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, params, symm_so_far):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"] // config["NUM_PRETRAINED"], env.observation_space(env.agents[0]).n)
            ),
            jnp.zeros((1, config["NUM_ENVS"] // config["NUM_PRETRAINED"])),
            jnp.zeros((1, config["NUM_ENVS"] // config["NUM_PRETRAINED"], env.action_space(env.agents[0]).n)),
            0
        )
        network_params = []
        for i, p in enumerate(params):
            net_p = network.init(_rng, init_x)
            updated_params = update_params(net_p, p)
            if symm_so_far is None:
                if i == 0:
                    in_permutation_layer_1 = updated_params['params']['Dense_0']
                    in_permutation_layer_2 = updated_params['params']['Dense_1']
                    in_permutation_layer_3 = updated_params['params']['Dense_2']
                    out_permutation = updated_params['params']['Dense_6']['kernel']
                else:
                    updated_params['params']['Dense_0'] = in_permutation_layer_1
                    updated_params['params']['Dense_1'] = in_permutation_layer_2
                    updated_params['params']['Dense_2'] = in_permutation_layer_3
                    updated_params['params']['Dense_6']['kernel'] = out_permutation
            else:
                updated_params['params']['Dense_0'] = symm_so_far[0]
                updated_params['params']['Dense_1'] = symm_so_far[1]
                updated_params['params']['Dense_2'] = symm_so_far[2]
                updated_params['params']['Dense_6']['kernel'] = symm_so_far[3]
                in_permutation_layer_1 = symm_so_far[0]
                in_permutation_layer_2 = symm_so_far[1]
                in_permutation_layer_3 = symm_so_far[2]
                out_permutation = symm_so_far[3]
            network_params.append(updated_params)
        print('Loaded')
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_states = []
        for net_p in network_params:
            train_states.append(
                TrainState.create(
                apply_fn=network.apply,
                params=net_p,
                tx=tx,
                )
            )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsvs = []
        env_states = []
        for _ in range(config["NUM_PRETRAINED"]):
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"] // config["NUM_PRETRAINED"])
            obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
            obsvs.append(obsv)
            env_states.append(env_state)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps, best_score, best_transformation = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, rng, num_compositions = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"] // config["NUM_PRETRAINED"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"] // config["NUM_PRETRAINED"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :], num_compositions)
                pi, value = network.apply(train_state.params, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"] // config["NUM_PRETRAINED"], env.num_agents)

                env_act = jax.tree_util.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"] // config["NUM_PRETRAINED"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_util.tree_map(lambda x: x.reshape((config["NUM_ACTORS"] // config["NUM_PRETRAINED"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"] // config["NUM_PRETRAINED"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"] // config["NUM_PRETRAINED"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions
                )
                runner_state = [train_state, env_state, obsv, done_batch, rng, num_compositions]
                return runner_state, transition

            runner_states = []
            traj_batches = []
            at_least_one_single_composition = False
            for i in range(config["NUM_PRETRAINED"]):
                rs = [runner_state[0][i], runner_state[1][i], runner_state[2][i], runner_state[3][i,:], runner_state[4]]

                _, _rng = jax.random.split(runner_state[4])
                probability = 0.5#0.05 + 0.2 * jnp.clip(update_steps / 1000, 0, 1)
                coin_flip = jax.random.uniform(_rng, ()) < probability
                num_compositions = jax.lax.cond(coin_flip, lambda x: 2, lambda x: 1, None)
                at_least_one_single_composition = jax.lax.cond(num_compositions == 1, lambda x: True, lambda x: x, at_least_one_single_composition)

                rs.append(num_compositions)

                rs, traj_batch = jax.lax.scan(
                    _env_step, rs, None, config["NUM_STEPS"]
                )
                runner_states.append(rs)
                traj_batches.append(traj_batch)

            accumulated_metrics = {
                'returned_episode': None,
                'returned_episode_lengths': None,
                'returned_episode_returns': None
            }

            for i, rs in enumerate(runner_states):
                traj_batch = traj_batches[i]
                train_state, env_state, last_obs, last_done, rng, num_compositions = rs
                if i > 0:
                    train_state.params['params']['Dense_0'] = runner_states[i-1][0].params['params']['Dense_0']
                    train_state.params['params']['Dense_1'] = runner_states[i-1][0].params['params']['Dense_1']
                    train_state.params['params']['Dense_2'] = runner_states[i-1][0].params['params']['Dense_2']
                    train_state.params['params']['Dense_6']['kernel'] = runner_states[i-1][0].params['params']['Dense_6']['kernel']

                last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"] // config["NUM_PRETRAINED"])
                avail_actions = jnp.ones(
                    (config["NUM_ACTORS"] // config["NUM_PRETRAINED"], env.action_space(env.agents[0]).n)
                )
                ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions, num_compositions)
                _, last_val = network.apply(train_state.params, ac_in)
                last_val = last_val.squeeze()

                # CALCULATE ADVANTAGE
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

                advantages, targets = _calculate_gae(traj_batch, last_val)

                # UPDATE NETWORK
                def _update_epoch(update_state, unused):
                    def _update_minbatch(train_state, batch_info):
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            # RERUN NETWORK
                            pi, value = network.apply(params,
                                                    (traj_batch.obs, traj_batch.done, traj_batch.avail_actions, num_compositions))
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
                            return total_loss, (loss_actor, value_loss, entropy)

                        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                        total_loss, grads = grad_fn(
                            train_state.params, traj_batch, advantages, targets
                        )
                        processed_grads = selective_zero_grads(grads)
                        train_state = train_state.apply_gradients(grads=processed_grads)
                        return train_state, total_loss

                    train_state, traj_batch, advantages, targets, rng = update_state
                    rng, _rng = jax.random.split(rng)

                    batch = (traj_batch, advantages.squeeze(), targets.squeeze())
                    permutation = jax.random.permutation(_rng, config["NUM_ACTORS"] // config["NUM_PRETRAINED"])

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
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(
                    _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
                )
                train_state = update_state[0]
                runner_states[i][0] = train_state
                metric = traj_batch.info
                rng = update_state[-1]
                for key in metric.keys():
                    if accumulated_metrics[key] is None:
                        # First iteration: initialize the array
                        accumulated_metrics[key] = metric[key][jnp.newaxis, ...]
                    else:
                        # Subsequent iterations: append the new data
                        accumulated_metrics[key] = jnp.concatenate([accumulated_metrics[key], metric[key][jnp.newaxis, ...]], axis=0)

            runner_states[0][0].params['params']['Dense_0'] = runner_states[-1][0].params['params']['Dense_0']
            runner_states[0][0].params['params']['Dense_1'] = runner_states[-1][0].params['params']['Dense_1']
            runner_states[0][0].params['params']['Dense_2'] = runner_states[-1][0].params['params']['Dense_2']
            runner_states[0][0].params['params']['Dense_6']['kernel'] = runner_states[-1][0].params['params']['Dense_6']['kernel']

            def callback(metric):
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )
            
            accumulated_metrics["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, accumulated_metrics)
            update_steps = update_steps + 1

            temp_runner_state = [item for item in runner_state]
            for i in range(config["NUM_PRETRAINED"]):
                temp_runner_state[0][i] = runner_states[i][0]
                temp_runner_state[1][i] = runner_states[i][1]
                temp_runner_state[2][i] = runner_states[i][2]
                temp_runner_state[3] = temp_runner_state[3].at[i, :].set(runner_states[i][3])
            temp_runner_state[4] = rng
            runner_state = tuple(temp_runner_state)

            def update_best_state(best_state, current_state, current_score):
                current_best_score, current_best_train_state = best_state
                return jax.lax.cond(
                    jnp.logical_and(current_score >= current_best_score, at_least_one_single_composition),
                    lambda: (current_score, current_state),
                    lambda: (current_best_score, current_best_train_state)
                )

            best_score, best_transformation = update_best_state(
                (best_score, best_transformation), 
                (runner_states[0][0].params['params']['Dense_0'], runner_states[0][0].params['params']['Dense_1'], runner_states[0][0].params['params']['Dense_2']), 
                accumulated_metrics["returned_episode_returns"][-1, :].mean()
            )

            return (runner_state, update_steps, best_score, best_transformation), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_states, env_states, obsvs, jnp.zeros((config["NUM_PRETRAINED"], config["NUM_ACTORS"] // config["NUM_PRETRAINED"]), dtype=bool), _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0, 0., (in_permutation_layer_1, in_permutation_layer_2, in_permutation_layer_3)), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="group_ippo_ff_hanabi_neural_symm_disc")
def main(config):
    config = OmegaConf.to_container(config) 

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    params = []
    for checkpoint in config["LOAD_CHECKPOINT"].split(','):
        params.append(load_checkpoint(checkpoint))
    config["NUM_PRETRAINED"] = len(params)

    symm_so_far = None

    for i in range(config["NUM_CHECKPOINTS"]):
        a = np.random.randint(9999999)
        print(a)
        rng = jax.random.PRNGKey(a)
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng, params, symm_so_far)
        print(jax.device_get(out["runner_state"][-1]))
        print(jax.device_get(out["runner_state"][0][0][0]).params['params']['Dense_6']['kernel'])
        print(jax.device_get(out["runner_state"][-2]))
        symm_so_far = [out["runner_state"][0][0][0].params['params']['Dense_0'], out["runner_state"][0][0][0].params['params']['Dense_1'], out["runner_state"][0][0][0].params['params']['Dense_2'], out["runner_state"][0][0][0].params['params']['Dense_6']['kernel']]
        save_permutation((jax.device_get(out["runner_state"][0][0][0]).params['params']['Dense_0'], jax.device_get(out["runner_state"][0][0][0]).params['params']['Dense_1'], jax.device_get(out["runner_state"][0][0][0]).params['params']['Dense_2']), 
                            jax.device_get(out["runner_state"][0][0][0]).params['params']['Dense_6']['kernel'], 
                            config, 
                            i+1)

if __name__ == "__main__":
    main()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''