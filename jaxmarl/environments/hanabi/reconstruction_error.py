import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import pickle
import random
import distrax
from typing import Sequence, NamedTuple, Any, Dict
import numpy as np
from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL as HanabiGame

# Helper functions
batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])

def get_neural_symmetry(path):
    with open(f'{path}ippo_symm_disc/in_permutation_layer_1_kernel_checkpoint_1.pkl', 'rb') as file:
        layer_1_kernel = pickle.load(file)
    with open(f'{path}ippo_symm_disc/in_permutation_layer_1_bias_checkpoint_1.pkl', 'rb') as file:
        layer_1_bias = pickle.load(file)
    with open(f'{path}ippo_symm_disc/in_permutation_layer_2_kernel_checkpoint_1.pkl', 'rb') as file:
        layer_2_kernel = pickle.load(file)
    with open(f'{path}ippo_symm_disc/in_permutation_layer_2_bias_checkpoint_1.pkl', 'rb') as file:
        layer_2_bias = pickle.load(file)
    with open(f'{path}ippo_symm_disc/in_permutation_layer_3_kernel_checkpoint_1.pkl', 'rb') as file:
        layer_3_kernel = pickle.load(file)
    with open(f'{path}ippo_symm_disc/in_permutation_layer_3_bias_checkpoint_1.pkl', 'rb') as file:
        layer_3_bias = pickle.load(file)
    with open(f'{path}ippo_symm_disc/out_permutation_checkpoint_1.pkl', 'rb') as file:
        out_permutation = pickle.load(file)

    return layer_1_kernel, layer_1_bias, layer_2_kernel, layer_2_bias, layer_3_kernel, layer_3_bias, out_permutation

def ippo_load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    if 'layer1' in params_dict['params']:
        key_map = {'layer1': 'Dense_0', 'layer2': 'Dense_1', 'layer3': 'Dense_2', 'layer4': 'Dense_3', 'layer5': 'Dense_4'}
        params_dict['params'] = {key_map[key]: value for key, value in params_dict['params'].items() if key in key_map}
    return params_dict

class IPPOAgent:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)
        self.model = ActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        pi, value = self.model.apply(self.params, (obs, done, legal_moves))
        return pi.sample(seed=rng)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)
        pi = distrax.Categorical(logits=action_logits)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)

# Environment setup
env = HanabiGame()
rng = jax.random.PRNGKey(99)
obs, env_state = env.reset(rng)
legal_moves = env.get_legal_moves(env_state)

# Create agents
agents = []
agents.append(IPPOAgent('/workspace/models/hanabi/new/ippo_ff_8/checkpoint_1.pkl', 0))
agents.append(IPPOAgent('/workspace/models/hanabi/new/ippo_ff_8/checkpoint_1.pkl', 1))

# Load neural symmetry
layer_1_kernel, layer_1_bias, layer_2_kernel, layer_2_bias, layer_3_kernel, layer_3_bias, out_permutation = get_neural_symmetry('/workspace/models/hanabi/new/mild-glade/')
# layer_1_kernel, layer_1_bias, layer_2_kernel, layer_2_bias, layer_3_kernel, layer_3_bias, out_permutation = get_neural_symmetry('/workspace/models/hanabi/neural_colour_1_colour_2_6pre_transpose_0/')

done = False
t = 0
total_reward = 0
reconstruction_error = 0

# Main loop
while not done:
    rng, _rng = jax.random.split(rng)
    curr_player = jnp.argmax(env_state.cur_player_idx)

    # Get actions for each agent
    actions_all = [agents[i].act(obs, done, legal_moves, curr_player, _rng) for i in range(len(env.agents))]

    # Determine which agent's action to use based on the current player
    if curr_player == 1:
        actions = {agent: jnp.array(actions_all[1][i]) for i, agent in enumerate(env.agents)}
    else:
        actions = {agent: jnp.array(actions_all[0][i]) for i, agent in enumerate(env.agents)}

    # Step the environment
    rng, _rng = jax.random.split(rng)
    obs, env_state, reward, done, _ = env.step(_rng, env_state, actions)
    legal_moves = env.get_legal_moves(env_state)

    # Compute reconstruction error
    if curr_player == 1:
        transformed_obs = nn.relu(obs['agent_0'] @ layer_1_kernel + layer_1_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_2_kernel + layer_2_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_3_kernel + layer_3_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_1_kernel + layer_1_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_2_kernel + layer_2_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_3_kernel + layer_3_bias)
        reconstruction_error += jnp.sqrt(jnp.sum((obs['agent_0'] - transformed_obs)**2)) / jnp.sqrt(jnp.sum(obs['agent_0']**2))
    else:
        transformed_obs = nn.relu(obs['agent_1'] @ layer_1_kernel + layer_1_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_2_kernel + layer_2_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_3_kernel + layer_3_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_1_kernel + layer_1_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_2_kernel + layer_2_bias)
        transformed_obs = nn.relu(transformed_obs @ layer_3_kernel + layer_3_bias)
        reconstruction_error += jnp.sqrt(jnp.sum((obs['agent_1'] - transformed_obs)**2)) / jnp.sqrt(jnp.sum(obs['agent_0']**2))

    # Update rewards and time steps
    total_reward += reward["__all__"]
    t += 1
    done = done["__all__"]

# Output final results
print(f'Total reward: {total_reward}')
print(f'Time steps: {t}')
print(f'Average reconstruction error: {reconstruction_error / t if t > 0 else 0}')