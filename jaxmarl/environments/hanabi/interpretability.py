import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from jaxmarl import make
from typing import Sequence, NamedTuple, Any, Dict
import distrax
import pickle
import random
import sys
import numpy as np
import argparse
from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL as HanabiGame

env = HanabiGame()
batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent: x[i] for i, agent in enumerate(env.agents)}

def ippo_load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    if 'layer1' in params_dict['params']:
        key_map = {'layer1': 'Dense_0', 'layer2': 'Dense_1', 'layer3': 'Dense_2', 'layer4': 'Dense_3', 'layer5': 'Dense_4'}
        params_dict['params'] = {key_map[key]: value for key, value in params_dict['params'].items() if key in key_map}
    return params_dict

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):
        indices = [5,6,7,8,9,0,1,2,3,4,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,30,31,32,33,34,25,26,27,28,29,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,55,56,57,58,59,50,51,52,53,54,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,80,81,82,83,84,75,76,77,78,79,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,105,106,107,108,109,100,101,102,103,104,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,
        127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,172,173,174,175,176,167,168,169,170,171,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,
        213,214,215,216,217,218,219,220,221,222,203,204,205,206,207,208,209,210,211,212,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,
        253,254,255,256,257,258,259,260,262,261,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,286,287,288,289,290,281,282,283,284,285,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,
        313, 314, 315, 316, 317, 308, 309, 310, 311, 312, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 334, 333, 335, 336, 337, 338, 339, 340, 341, 342, 348, 349, 350, 351, 352, 343, 344, 345, 346, 347, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 369, 368, 370, 371, 372, 373, 374, 375, 376, 377, 383, 384, 385, 386, 387, 378, 379, 380, 381, 382, 388, 389, 390, 
        391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 404, 403, 405, 406, 407, 408, 409, 410, 411, 412, 418, 419, 420, 421, 422, 413, 414, 415, 416, 417, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 439, 438, 440, 441, 442, 443, 444, 445, 446, 447, 453, 454, 455, 456, 457, 448, 449, 450, 451, 452, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 474, 
        473, 475, 476, 477, 478, 479, 480, 481, 482, 488, 489, 490, 491, 492, 483, 484, 485, 486, 487, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 509, 508, 510, 511, 512, 513, 514, 515, 516, 517, 523, 524, 525, 526, 527, 518, 519, 520, 521, 522, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 544, 543, 545, 546, 547, 548, 549, 550, 551, 552, 558, 559, 560, 561, 
        562, 553, 554, 555, 556, 557, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 579, 578, 580, 581, 582, 583, 584, 585, 586, 587, 593, 594, 595, 596, 597, 588, 589, 590, 591, 592, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 614, 613, 615, 616, 617, 618, 619, 620, 621, 622, 628, 629, 630, 631, 632, 623, 624, 625, 626, 627, 633, 634, 635, 636, 637, 638, 639, 
        640, 641, 642, 643, 644, 645, 646, 647, 649, 648, 650, 651, 652, 653, 654, 655, 656, 657]
        self.P_in = jnp.zeros((658, 658), dtype=jnp.float32)
        for i in range(658):
            self.P_in = self.P_in.at[i, indices[i]].set(1.)

        self.P_out = jnp.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

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

class TransformedObsIPPOAgent:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)
        self.model = ActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng):
        obs = batchify(obs)

        legal_moves = batchify(legal_moves)
        pi, value = self.model.apply(self.params, (obs, done, legal_moves))
        return pi.sample(seed=rng)

def get_agents(args):
    agents = []
    for player_idx in [0, 1]:
        player_type = getattr(args, f"player{player_idx}")
        weight_file = getattr(args, f"weight{player_idx}")
        if player_type == "ippo":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            agents.append(IPPOAgent(weight_file, player_idx))
        elif player_type == "transformed_obs_ippo":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            agents.append(IPPOAgent(weight_file, player_idx))
    return agents

def play_game(rng, args):

    agents = get_agents(args)

    use_jit = args.use_jit if args.use_jit is not None else True
    with jax.disable_jit(not use_jit):

        rng, _rng = jax.random.split(rng)

        obs, env_state = env.reset(_rng)
        legal_moves = env.get_legal_moves(env_state)

        @jax.jit
        def _step_env(rng, env_state, actions):
            rng, _rng = jax.random.split(rng)
            new_obs, new_env_state, reward, dones, infos = env.step(
                _rng, env_state, actions
            )
            new_legal_moves = env.get_legal_moves(new_env_state)
            return rng, new_env_state, new_obs, reward, dones, new_legal_moves

        done = False
        cum_rew = 0
        t = 0

        def cond_fn(val):
            _, _, done, _, _, _, _, _, _ = val
            return jax.numpy.logical_not(done)

        def body_fn(val):
            cum_rew, t, done, rng, env_state, obs, legal_moves, conditional_action_dist, last_action = val
            rng, _rng = jax.random.split(rng)
            curr_player = jnp.argmax(env_state.cur_player_idx)
            actions_all = [
                agents[i].act(obs, done, legal_moves, curr_player, _rng)
                for i in range(len(env.agents))
            ]

            def true_fn(_):
                return {
                    agent: jnp.array(actions_all[1][i]) for i, agent in enumerate(env.agents)
                }

            def false_fn(_):
                return {
                    agent: jnp.array(actions_all[0][i]) for i, agent in enumerate(env.agents)
                }

            actions = jax.lax.cond(curr_player == 1, true_fn, false_fn, None)

            def curr_action_1(_):
                return actions['agent_1']

            def curr_action_0(_):
                return actions['agent_0']

            curr_action = jax.lax.cond(curr_player == 1, curr_action_1, curr_action_0, None)

            update = jnp.zeros_like(conditional_action_dist)
            update = update.at[last_action, curr_action].set(1)
            conditional_action_dist = conditional_action_dist + update

            last_action = curr_action

            rng, env_state, obs, reward, dones, legal_moves = _step_env(
                rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew += reward["__all__"]
            t += 1
            return (cum_rew, t, done, rng, env_state, obs, legal_moves, conditional_action_dist, last_action)

        conditional_action_dist = jnp.zeros((21,21))

        last_action = 20
        init_val = (0, 0, False, rng, env_state, obs, legal_moves, conditional_action_dist, last_action)
        cum_rew, _, _, _, _, _, _, conditional_action_dist, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

        return(conditional_action_dist)

def play_games(rng_keys, args):
    play_game_vectorized = jax.vmap(play_game, in_axes=(0, None))
    results = play_game_vectorized(rng_keys, args)
    return results


def main(args):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)

    rng = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng, args.num_rollouts)

    action_dist = play_games(rng_keys, args)
    action_dist = jnp.sum(action_dist, axis=0)
    np.save('conditional_dist.npy', action_dist)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player0", type=str, default="ippo")
    parser.add_argument("--player1", type=str, default="ippo")
    parser.add_argument("--weight0", type=str, default="/workspace/models/hanabi/new/realop_3/op_ippo_ff/checkpoint_5.pkl")
    parser.add_argument("--weight1", type=str, default="/workspace/models/hanabi/new/realop_3/op_ippo_ff/checkpoint_5.pkl")
    # parser.add_argument("--weight0", type=str, default="/workspace/models/hanabi/neural_op_1_6pre_0/op_ippo_ff/checkpoint_5.pkl")
    # parser.add_argument("--weight1", type=str, default="/workspace/models/hanabi/neural_op_1_6pre_0/op_ippo_ff/checkpoint_5.pkl")
    # parser.add_argument("--weight0", type=str, default="/workspace/models/hanabi/neural_op_6pre_0_diff_pi_5/op_ippo_ff/checkpoint_5.pkl")
    # parser.add_argument("--weight1", type=str, default="/workspace/models/hanabi/neural_op_6pre_0_diff_pi_5/op_ippo_ff/checkpoint_5.pkl")
    # parser.add_argument("--weight0", type=str, default="/workspace/models/hanabi/neural_op_6pre_0_15phi_diff_pi_5/op_ippo_ff/checkpoint_9.pkl")
    # parser.add_argument("--weight1", type=str, default="/workspace/models/hanabi/neural_op_6pre_0_15phi_diff_pi_5/op_ippo_ff/checkpoint_9.pkl")
    # parser.add_argument("--weight0", type=str, default="/workspace/models/hanabi/new/op-sage-wave/op_ippo_ff/checkpoint_7.pkl")
    # parser.add_argument("--weight1", type=str, default="/workspace/models/hanabi/new/op-sage-wave/op_ippo_ff/checkpoint_7.pkl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_rollouts", type=int, default=5000)
    parser.add_argument("--use_jit", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)