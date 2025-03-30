"""
Based on PureJaxRL Implementation of PPO
"""

import os
import pickle

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict, List
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

def save_checkpoint(params, step, checkpoint_path='/workspace/hanabi_saves/op_ippo_ff/'):
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_{step}.pkl")

    params_dict = jax.tree_map(lambda x: np.array(x), params)

    with open(checkpoint_file, 'wb') as f:
        pickle.dump(params_dict, f)

def load_permutation(path, i):
    with open(f'{path}symmetry_{i}_in_2p.pkl', 'rb') as file:
        in_permutation = pickle.load(file)
    with open(f'{path}symmetry_{i}_out_2p.pkl', 'rb') as file:
        out_permutation = pickle.load(file)
    return in_permutation, out_permutation

def identity_init():
    def init(key, shape, dtype=jnp.float32):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("Shape must be square for a permutation matrix, and act on 21 dimensional action space.")
        return jnp.eye(shape[0], dtype=dtype)
    return init

class DenseLayerConfig(NamedTuple):
    units: int
    kernel: Any

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        self.layer1 = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.layer2 = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.layer3 = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))
        self.layer4 = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))
        self.layer5 = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(self, x):
        obs, dones, avail_actions, out_permutations = x

        embedding = self.layer1(obs)
        embedding = nn.relu(embedding)

        actor_mean = self.layer2(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = self.layer3(actor_mean)

        transformed_actor_mean = jax.vmap(lambda a, p: jnp.dot(a, p))(actor_mean.reshape(-1,actor_mean.shape[-1]), out_permutations).reshape(actor_mean.shape)

        unavail_actions = 1 - avail_actions
        action_logits = transformed_actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = self.layer4(embedding)
        critic = nn.relu(critic)
        critic = self.layer5(critic)

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
    shuffle_colour_indices: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def make_train(config):
    # env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = HanabiGame(num_agents=3)
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

    def train(rng, params, in_permutations, out_permutations):

        # INIT NETWORK
        network = ActorCritic(env.action_space(env.agents[0]).n, config=config)
        
        rng, _rng, __rng = jax.random.split(rng, 3)
        shuffle_colour_indices = jax.random.choice(__rng, in_permutations.shape[0], shape=(config["NUM_ENVS"],), replace=True)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], env.observation_space(env.agents[0]).n)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
            jnp.zeros((1, config["NUM_ENVS"], env.action_space(env.agents[0]).n)),
            out_permutations[shuffle_colour_indices]
        )
       
        if params is None:
            network_params = network.init(_rng, init_x)
        else:
            network_params = params

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng, __rng = jax.random.split(rng, 3)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng)
        shuffle_colour_indices = jax.random.choice(__rng, in_permutations.shape[0], shape=(config["NUM_ACTORS"],), replace=True)

        def transform_obs(obs, in_permutation):
            transformed_obs = jnp.dot(obs, in_permutation)
            return transformed_obs

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                avail_actions = jax.vmap(env.get_legal_moves)(env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents, config["NUM_ACTORS"])
                )
                obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])

                obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                    obs_batch.reshape(-1, obs_batch.shape[-1]),
                    in_permutations[shuffle_colour_indices]
                ).reshape(obs_batch.shape)

                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :], avail_actions[np.newaxis, :], out_permutations[shuffle_colour_indices])
                pi, value = network.apply(train_state.params, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents)

                env_act = jax.tree_map(lambda x: x.squeeze(), env_act)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(
                    rng_step, env_state, env_act
                )
                info = jax.tree_map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                
                def _resample_indices(rng, shuffle_colour_indices, done_batch, max_index):
                    rng, _rng = jax.random.split(rng)

                    new_indices = jax.random.choice(_rng, max_index, shape=shuffle_colour_indices.shape, replace=True)

                    updated_shuffle_colour_indices = jnp.where(done_batch, new_indices, shuffle_colour_indices)

                    return rng, updated_shuffle_colour_indices

                rng, updated_shuffle_colour_indices = _resample_indices(rng, shuffle_colour_indices, done_batch, in_permutations.shape[0])

                transition = Transition(
                    done_batch,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    info,
                    avail_actions,
                    shuffle_colour_indices
                )
                runner_state = (train_state, env_state, obsv, done_batch, updated_shuffle_colour_indices, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ACTORS"])
            last_obs_batch = jax.vmap(transform_obs, in_axes=(0, 0))(
                    last_obs_batch.reshape(-1, last_obs_batch.shape[-1]),
                    in_permutations[shuffle_colour_indices],
            ).reshape(last_obs_batch.shape)
            avail_actions = jnp.ones(
                (config["NUM_ACTORS"], env.action_space(env.agents[0]).n)
            )
            ac_in = (
                last_obs_batch[np.newaxis, :], 
                last_done[np.newaxis, :], 
                avail_actions, 
                out_permutations[shuffle_colour_indices]
            )
            _, last_val = network.apply(train_state.params, ac_in)
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
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK

                        pi, value = network.apply(params,
                                                  (traj_batch.obs, traj_batch.done, traj_batch.avail_actions, out_permutations[traj_batch.shuffle_colour_indices.ravel()]))
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
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch = (traj_batch, advantages.squeeze(), targets.squeeze())
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
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
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
            runner_state = (train_state, env_state, last_obs, last_done, shuffle_colour_indices, rng)
            return (runner_state, update_steps), None

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ACTORS"]), dtype=bool), shuffle_colour_indices, _rng)
        runner_state, _ = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

@hydra.main(version_base=None, config_path="config", config_name="decpomdp_symmetries_op_ippo_ff_hanabi")
def main(config):
    config = OmegaConf.to_container(config) 

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["IPPO", "FF", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    in_permutations = jnp.zeros((120, 658, 658))
    out_permutations = jnp.zeros((120, 21, 21))
    for i in range(120):
        in_permutation, out_permutation = load_permutation(config["LOAD_PERMS"], i)
        in_permutations = in_permutations.at[i, :].set(in_permutation)
        out_permutations = out_permutations.at[i, :].set(out_permutation)
    config["NUM_PERMUTATIONS"] = in_permutations.shape[0]

    params = None
    for i in range(config["NUM_CHECKPOINTS"]):
        rng = jax.random.PRNGKey(np.random.randint(9999999))
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng, params, in_permutations, out_permutations)
        params = out["runner_state"][0][0].params
        save_checkpoint(jax.device_get(out["runner_state"][0][0]).params, i+1)


if __name__ == "__main__":
    main()
    '''results = out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1)
    jnp.save('hanabi_results', results)
    plt.plot(results)
    plt.xlabel("Update Step")
    plt.ylabel("Return")
    plt.savefig(f'IPPO_{config["ENV_NAME"]}.png')'''