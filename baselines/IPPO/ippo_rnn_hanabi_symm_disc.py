"""
Based on PureJaxRL Implementation of PPO
"""
import pickle
import os

import jax
import jax.numpy as jnp
import flax.linen as nn
import flax.traverse_util as traverse_util
import numpy as np
from scipy.optimize import linear_sum_assignment
import optax
from jax.nn.initializers import Initializer as Initializer
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper, FlattenObservationWrapper
# import jaxmarl
# from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL as HanabiGame
import wandb
import functools
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf

def nearest_permutation_matrix(A):
    n = A.shape[0]
    # Expanding A and the identity matrix for broadcasting
    A_expanded = A[:, np.newaxis, :]  # Shape (n, 1, n)
    identity_expanded = np.eye(n)[np.newaxis, :, :]  # Shape (1, n, n)

    # Vectorized computation of the cost matrix
    cost_matrix = np.sum((A_expanded - identity_expanded) ** 2, axis=2)

    # Applying the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Constructing the permutation matrix
    P = np.zeros_like(A)
    P[row_ind, col_ind] = 1

    return P

def identity_init(scale=1.0):
    def init(key, shape, dtype=jnp.float32):
        # Ensure the shape is for a square matrix
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initialization requires a square matrix shape")
        return scale * jnp.eye(shape[0], dtype=dtype)
    return init

def identity_jitter_init(scale=1.0, jitter=0.01):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] != shape[1]:
            raise ValueError("Identity matrix initialization requires a square matrix shape")
        # Create the identity matrix scaled by the factor
        identity = scale * jnp.eye(shape[0], dtype=dtype)
        # Create jitter noise of the same shape as the identity matrix
        noise = jitter * jax.random.normal(key, shape, dtype=dtype)
        return identity + noise
    return init

def random_doubly_stochastic_init(scale=1.0, iterations=10000):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] != shape[1]:
            raise ValueError("Doubly stochastic matrix initialization requires a square matrix shape")
        # Initialize with random values
        matrix = jax.random.uniform(key, shape, dtype=dtype)

        # Sinkhorn's iteration
        for _ in range(iterations):
            # Normalize rows
            matrix /= matrix.sum(axis=1, keepdims=True)
            # Normalize columns
            matrix /= matrix.sum(axis=0, keepdims=True)

        return scale * matrix

    return init

def biased_permutation_matrix(key, size, bias_strength=0.5):
    # Start with the identity matrix
    permutation = jnp.eye(size)

    # Determine the number of swaps to perform based on the bias strength
    num_swaps = int((1 - bias_strength) * size)

    for _ in range(num_swaps):
        # Generate two random indices to swap
        i, j = jax.random.choice(key, size, shape=(2,), replace=False)

        # Perform the swap
        temp = permutation[i].copy()
        permutation = permutation.at[i].set(permutation[j])
        permutation = permutation.at[j].set(temp)

    return permutation


def biased_doubly_stochastic_init(bias_strength=0.5, permutation_bias=0.8, iterations=100):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] != shape[1]:
            raise ValueError("Doubly stochastic matrix initialization requires a square matrix shape")

        key1, key2 = jax.random.split(key)

        # Generate a biased permutation matrix
        perm = biased_permutation_matrix(key1, shape[0], bias_strength=permutation_bias)

        # Start with random values
        matrix = jax.random.uniform(key2, shape, dtype=dtype)

        # Sinkhorn's iteration to make it doubly stochastic
        for _ in range(iterations):
            matrix /= matrix.sum(axis=1, keepdims=True)  # Normalize rows
            matrix /= matrix.sum(axis=0, keepdims=True)  # Normalize columns

        # Add bias towards the biased permutation matrix after Sinkhorn's iteration
        matrix = matrix * (1 - bias_strength) + perm * bias_strength

        return matrix

    return init

def biased_random_init(bias_strength=0.5, permutation_bias=0.8, minval=-1, maxval=1):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] != shape[1]:
            raise ValueError("Doubly stochastic matrix initialization requires a square matrix shape")

        key1, key2 = jax.random.split(key)

        # Generate a biased permutation matrix
        perm = biased_permutation_matrix(key1, shape[0], bias_strength=permutation_bias)

        # Start with random values
        matrix = jax.random.uniform(key2, shape, dtype=dtype, minval=minval, maxval=maxval)

        # Add bias towards the biased permutation matrix after Sinkhorn's iteration
        matrix = matrix * (1 - bias_strength) + perm * bias_strength

        return matrix

    return init

def biased_orthogonal_init(bias_strength=0.5, permutation_bias=0.8, orthogonal_scale=np.sqrt(2)):
    def init(key, shape, dtype=jnp.float32):
        if shape[0] != shape[1]:
            raise ValueError("Doubly stochastic matrix initialization requires a square matrix shape")

        key1, key2 = jax.random.split(key)

        # Generate a biased permutation matrix
        perm = biased_permutation_matrix(key1, shape[0], bias_strength=permutation_bias)

        # Generate orthogonal matrix
        orthogonal_matrix = orthogonal_scale * orthogonal()(key, shape, dtype)

        # Add bias towards the biased permutation matrix after Sinkhorn's iteration
        matrix = orthogonal_matrix * (1 - bias_strength) + perm * bias_strength

        return matrix

    return init

def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    return params_dict

def save_checkpoint(params, config, step):
    os.makedirs(config["SAVE_CHECKPOINT"], exist_ok=True)
    checkpoint_file = os.path.join(config["SAVE_CHECKPOINT"], f"checkpoint_{step}.pkl")

    params_dict = jax.tree_map(lambda x: np.array(x), params)

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(params_dict, f)

def update_params(init_params, loaded_params):
    # Flatten both parameter trees
    flat_init_params = traverse_util.flatten_dict(init_params)
    flat_loaded_params = traverse_util.flatten_dict(loaded_params)

    mapping = {
        'Dense_1': 'Dense_0',
        'Dense_2': 'Dense_1',
        'Dense_3': 'Dense_2',
        'Dense_5': 'Dense_3',
        'Dense_6': 'Dense_4'
    }

    # Combine parameters, excluding the new layers
    combined_params = {}
    for key in flat_init_params.keys():
        mapped_key = mapping.get(key[1], key)
        if isinstance(mapped_key, str):
            mapped_key = (key[0], mapped_key, key[2])
        if mapped_key in flat_loaded_params and key[1] != 'Dense_0' and key[1] != 'Dense_4':
            combined_params[key] = flat_loaded_params[mapped_key]
        else:
            combined_params[key] = flat_init_params[key]

    # Unflatten to get the final parameters structure
    return traverse_util.unflatten_dict(combined_params)

def selective_zero_grads(grads, freeze_layers=('Dense_0', 'Dense_4')):
    """
    Recursively zero out gradients for all parameters except those in specified layers.
    This function handles both simple gradient arrays and nested structures.
    """

    def zero_grads_cond(grad, freeze):
        """ Helper function to conditionally zero out gradients. """
        return jax.lax.cond(freeze, lambda _: jnp.zeros_like(grad), lambda _: grad, None)

    def process_grads(item, layer):
        # Check if item is a gradient array or a nested dictionary
        if isinstance(item, dict):
            # Process each item in the nested dictionary
            return {k: process_grads(v, layer) for k, v in item.items()}
        else:
            # Apply zeroing condition to the gradient array
            freeze = not any(freeze_layer in layer for freeze_layer in freeze_layers)
            return zero_grads_cond(item, freeze)

    # Apply process_grads to each layer in the gradient dictionary
    processed_grads = {layer: process_grads(layer_grads, layer) for layer, layer_grads in grads['params'].items()}
    return {'params': processed_grads}

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
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        n = 783#obs.shape[-1]  # Size of the observation feature space
        # in_permutation = nn.Dense(n, kernel_init=random_doubly_stochastic_init(scale=1), use_bias=False)
        in_permutation = nn.Dense(n, kernel_init=biased_doubly_stochastic_init(bias_strength=self.config["IN_PERMUTATION_BIAS_STRENGTH"], permutation_bias=self.config["IN_PERMUTATION_PERM_BIAS"]), use_bias=False)
        # in_permutation = nn.Dense(n, kernel_init=orthogonal(np.sqrt(2)), use_bias=False)
        transformed_obs = in_permutation(obs) # Apply the linear layer to observations

        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(transformed_obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # out_permutation = nn.Dense(self.action_dim, kernel_init=random_doubly_stochastic_init(scale=1), use_bias=False)
        out_permutation = nn.Dense(self.action_dim, kernel_init=biased_doubly_stochastic_init(bias_strength=self.config["OUT_PERMUTATION_BIAS_STRENGTH"], permutation_bias=self.config["OUT_PERMUTATION_PERM_BIAS"]), use_bias=False)
        # out_permutation = nn.Dense(self.action_dim, kernel_init=orthogonal(np.sqrt(2)), use_bias=False)
        # out_permutation = nn.Dense(self.action_dim, kernel_init=identity_jitter_init(1), use_bias=False)
        transformed_actor_mean = out_permutation(actor_mean)

        unavail_actions = 1 - avail_actions
        action_logits = transformed_actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


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


def make_train(config, first):
    # env, env_params = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = HanabiGame()
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    if first:
        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS_FIRST"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
    else:
        config["NUM_UPDATES"] = (
                config["TOTAL_TIMESTEPS_SECOND"] // config["NUM_STEPS"] // config["NUM_ACTORS"] // config["NUM_PROJECTIONS"]
        )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # env = FlattenObservationWrapper(env) # NOTE need a batchify wrapper
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng, params, first):

        # INIT NETWORK
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n))
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], 128)
        network_params = network.init(_rng, init_hstate, init_x)
        if params is not None and first:
            updated_params = update_params(network_params, params)
            network_params = updated_params
            print('Loaded')
        elif params is not None:
            network_params = params
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5)
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ACTORS"], 128)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions

                )
                runner_state = (train_state, env_state, obsv, done_batch, hstate, rng)
                return runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (last_obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions)
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze()

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
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(params, init_hstate.transpose(),
                                                     (traj_batch.obs, traj_batch.done, traj_batch.avail_actions))
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

                        # DOUBLE STOCHASTIC REGULARIZATION
                        # in_row_sum = jnp.sum(jnp.abs(jnp.sum(params['params']['Dense_0']['kernel'], axis=1) - 1))
                        # in_col_sum = jnp.sum(jnp.abs(jnp.sum(params['params']['Dense_0']['kernel'], axis=0) - 1))

                        # out_row_sum = jnp.sum(jnp.abs(jnp.sum(params['params']['Dense_4']['kernel'], axis=1) - 1))
                        # out_col_sum = jnp.sum(jnp.abs(jnp.sum(params['params']['Dense_4']['kernel'], axis=0) - 1))
                        
                        # negative_mask_in = params['params']['Dense_0']['kernel'] < 0
                        # non_neg_in = jnp.sum(jnp.abs(jnp.where(negative_mask_in, params['params']['Dense_0']['kernel'], 0)))

                        # negative_mask_out = params['params']['Dense_4']['kernel'] < 0
                        # non_neg_out = jnp.sum(jnp.abs(jnp.where(negative_mask_out, params['params']['Dense_4']['kernel'], 0)))

                        # double_stoch_in_reg = in_row_sum + in_col_sum
                        # double_stoch_out_reg = out_row_sum + out_col_sum

                        # DISINCENTIVIZE IDENTITY CONVERGENCE
                        frob_norm_diff_in = jnp.sum(jnp.abs(params['params']['Dense_0']['kernel'] - jnp.eye(783)))
                        frob_norm_diff_out = jnp.sum(jnp.abs(params['params']['Dense_4']['kernel'] - jnp.eye(21)))

                        identity_penalty_in = 1 / (frob_norm_diff_in + 1e-3)
                        identity_penalty_out = 1 / (frob_norm_diff_out + 1e-3)

                        total_loss = (
                                loss_actor
                                + config["VF_COEF"] * value_loss
                                - config["ENT_COEF"] * entropy
                                # + config["DOUBLE_STOCH_IN_COEFF"] * double_stoch_in_reg
                                # + config["DOUBLE_STOCH_OUT_COEFF"] * double_stoch_out_reg
                                # + config["NON_NEG_IN_COEFF"] * non_neg_in
                                # + config["NON_NEG_OUT_COEFF"] * non_neg_out
                                + config["NON_IDENTITY_IN_COEFF"] * identity_penalty_in
                                + config["NON_IDENTITY_OUT_COEFF"] * identity_penalty_out
                        )
                        return total_loss, (value_loss, loss_actor, entropy, identity_penalty_in, identity_penalty_out) #double_stoch_in_reg, double_stoch_out_reg, non_neg_in, non_neg_out, identity_penalty_in, identity_penalty_out)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    processed_grads = selective_zero_grads(grads)
                    train_state = train_state.apply_gradients(grads=processed_grads)
                    return train_state, total_loss

                train_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                init_hstate = jnp.reshape(init_hstate, (config["NUM_STEPS"], config["NUM_ACTORS"]))
                batch = (init_hstate, traj_batch, advantages.squeeze(), targets.squeeze())
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
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            init_hstate = initial_hstate[None, :].squeeze().transpose()
            update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):           
                wandb.log(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * config["NUM_ENVS"]
                        * config["NUM_STEPS"],
                    }
                )
            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), init_hstate, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train


@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_hanabi_symm_disc")
def main(config):
    config = OmegaConf.to_container(config) 

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    params = load_checkpoint(config["LOAD_CHECKPOINT"])
    first = True
    a = np.random.randint(9999999)
    print(a)
    rng = jax.random.PRNGKey(a)
    train_jit = jax.jit(make_train(config, first), static_argnums=(2,))
    out = train_jit(rng, params, first)
    first = False
    params = out["runner_state"][0][0].params
    for i in range(config["NUM_PROJECTIONS"]):
        a = np.random.randint(9999999)
        print(a)
        rng = jax.random.PRNGKey(a)
        train_jit = jax.jit(make_train(config, first), static_argnums=(2,))
        out = train_jit(rng, params, first)
        params = jax.device_get(out["runner_state"][0][0]).params
        save_checkpoint(params, config, i+1)
        params['params']['Dense_0']['kernel'] = nearest_permutation_matrix(params['params']['Dense_0']['kernel'])
        params['params']['Dense_4']['kernel'] = nearest_permutation_matrix(params['params']['Dense_4']['kernel'])

if __name__ == "__main__":
    main()