import os
import jax
import abc
import struct
from jax import numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.linen.module import compact, nowrap
from flax import core, struct
from jaxmarl import make
from typing import Sequence, NamedTuple, Any, Dict, Callable, Tuple
import chex
import distrax
import pickle
import random
import sys
import numpy as np
import argparse
from jaxmarl.wrappers.baselines import load_params
from jaxmarl.environments.hanabi.hanabi_obl import HanabiOBL

env = HanabiOBL()
batchify = lambda x: jnp.stack([x[agent] for agent in env.agents])
unbatchify = lambda x: {agent: x[i] for i, agent in enumerate(env.agents)}

def load_permutation(path, i):
    with open(f'{path}symmetry_{i}_in.pkl', 'rb') as file:
        in_permutation = pickle.load(file)
    with open(f'{path}symmetry_{i}_out.pkl', 'rb') as file:
        out_permutation = pickle.load(file)
    return in_permutation, out_permutation

def ippo_load_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'rb') as f:
        params_dict = pickle.load(f)
    if 'layer1' in params_dict['params']:
        key_map = {'layer1': 'Dense_0', 'layer2': 'Dense_1', 'layer3': 'Dense_2', 'layer4': 'Dense_3', 'layer5': 'Dense_4'}
        params_dict['params'] = {key_map[key]: value for key, value in params_dict['params'].items() if key in key_map}
    return params_dict

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


class Agent(abc.ABC, struct.PyTreeNode):
    """
    Base class for agents that are to be evaluated in cross-play.

    You should implement this class for agents that you want to evaluate.

    The agent must be compatible with Jax transformations.

    Note:
        - Inherit from this class when implementing an agent.
        - Implement the `create` method for creating the agent.
        - Implement the `act` method for the agent to take an action given the observation and legal actions.
        - Ensure all methods use only JAX transformation compatible operations.
    """

    @classmethod
    def create(cls, *args, **kwargs):
        """
        Create and initialize an instance of the Agent.

        Implement this method to set up your agent with any necessary parameters or initial state.

        Returns:
            Agent: An instance of your implemented Agent class.
        """
        raise NotImplementedError("You must implement the 'create' method.")

    def act(self, observation: chex.Array, legal_actions: chex.Array) -> tuple[chex.Array, 'Agent']:
        """
        Determine the agent's action based on the current observation and legal actions.

        Implement this method to define your agent's policy.

        Args:
            observation (chex.Array): The current observation of the environment.
                Shape: (obs_dim,)
            legal_actions (chex.Array): A binary mask of legal actions.
                Shape: (action_dim,)

        Returns:
            Tuple[chex.Array, Agent]:
                - chex.Array: The selected action. Shape: () (scalar)
                - Agent: The updated agent state.

        Note:
            - This method will be jitted, so use only JAX operations.
            - The returned Agent should reflect any updates to the agent's state.
        """
        raise NotImplementedError("You must implement the 'act' method.")

class MultiLayerLSTM(nn.RNNCellBase):
    num_layers: int
    features: int

    @compact
    def __call__(self, carry, inputs):
        new_hs, new_cs = [], []
        y = inputs
        for layer in range(self.num_layers):
            new_carry, y = nn.LSTMCell(self.features, name=f"l{layer}")(
                jax.tree_map(lambda x: x[layer], carry), inputs
            )
            new_cs.append(new_carry[0])
            new_hs.append(new_carry[1])
            inputs = y

        new_final_carry = (jnp.stack(new_cs), jnp.stack(new_hs))
        return new_final_carry, y

    @nowrap
    def initialize_carry(self, rng: chex.PRNGKey, batch_dims: Tuple[int, ...]) -> Tuple[chex.Array, chex.Array]:
        mem_shape = (self.num_layers,) + batch_dims + (self.features,)
        c = jnp.zeros(mem_shape)
        h = jnp.zeros(mem_shape)
        return c, h

    @property
    def num_feature_axes(self) -> int:
        return 1

class OblR2D2(nn.Module):
    hid_dim: int = 512
    out_dim: int = 21
    num_lstm_layer: int = 2
    num_ff_layer: int = 1

    @compact
    def __call__(self, carry, inputs):
        priv_s, publ_s = inputs

        # Private network.
        priv_o = nn.Sequential(
            [
                nn.Dense(self.hid_dim, name="priv_net_dense_0"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_1"),
                nn.relu,
                nn.Dense(self.hid_dim, name="priv_net_dense_2"),
                nn.relu,
            ]
        )(priv_s)

        # Public network (MLP+lstm)
        x = nn.Sequential([nn.Dense(self.hid_dim, name="publ_net_dense_0"), nn.relu])(publ_s)
        carry, publ_o = MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim, name="lstm"
        )(carry, x)

        o = priv_o * publ_o
        a = nn.Dense(self.out_dim, name="fc_a")(o)
        return carry, a

    @nowrap
    def initialize_carry(self, rng: chex.PRNGKey, batch_dims: Tuple[int, ...]) -> Tuple[chex.Array, chex.Array]:
        return MultiLayerLSTM(
            num_layers=self.num_lstm_layer, features=self.hid_dim
        ).initialize_carry(rng, batch_dims)


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
        greedy_action = jnp.argmax(action_logits, axis=-1)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return greedy_action, jnp.squeeze(critic, axis=-1)

class TinActorCritic(nn.Module):
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
        greedy_action = jnp.argmax(action_logits, axis=-1)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return greedy_action, jnp.squeeze(critic, axis=-1)

class MDPSymmActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    def setup(self):

        colour_map = np.zeros(5)

        colour_map[0] = 0
        colour_map[1] = 1
        colour_map[2] = 3
        colour_map[3] = 2
        colour_map[4] = 4

        indices = np.zeros(658)

        for card in range(5):
            for colour in range(5):
                for rank in range(5):
                    indices[card * 25 + colour * 5 + rank] = card * 25 + colour_map[colour] * 5 + rank
                    
        indices[125] = 125
        indices[126] = 126

        curr_idx = 127

        for deck_feat in range(40):
            indices[curr_idx] = curr_idx
            curr_idx += 1
            
        for firework in range(5):
            for rank in range(5):
                indices[curr_idx + firework * 5 + rank] = curr_idx + colour_map[firework] * 5 + rank
                
        curr_idx += 25
                
        for token in range(11):
            indices[curr_idx] = curr_idx
            curr_idx += 1
            
        for discard_colour in range(5):
            for rank in range(10):
                indices[curr_idx + discard_colour * 10 + rank] = curr_idx + colour_map[discard_colour] * 10 + rank
                
        curr_idx += 50
                
        for more_last_action_feats in range(8):
            indices[curr_idx] = curr_idx
            curr_idx += 1
            
        for colour_revealed in range(5):
            indices[curr_idx + colour_revealed] = curr_idx + colour_map[colour_revealed]
            
        curr_idx += 5

        for more_last_action_feats in range(15):
            indices[curr_idx] = curr_idx
            curr_idx += 1
            
        for colour in range(5):
            for rank in range(5):
                indices[curr_idx + colour * 5 + rank] = curr_idx + colour_map[colour] * 5 + rank
                
        curr_idx += 25
            
        for more_last_action_feats in range(2):
            indices[curr_idx] = curr_idx
            curr_idx += 1
            
        for card in range(10):
            for colour in range(5):
                for rank in range(5):
                    indices[curr_idx + colour * 5 + rank] = curr_idx + colour_map[colour] * 5 + rank
            curr_idx += 25
            for colour in range(5):
                indices[curr_idx + colour] = curr_idx + colour_map[colour]
            curr_idx += 5
            for rank in range(5):
                indices[curr_idx] = curr_idx
                curr_idx += 1

        self.P_in = jnp.zeros((658, 658), dtype=jnp.float32)
        for i in range(658):
            self.P_in = self.P_in.at[i, int(indices[i])].set(1.)

        indices = np.zeros(21)
        for i in range(10):
            indices[i] = i
        for i in range(5):
            indices[i + 10] = colour_map[i] + 10
        for i in range(6):
            indices[i + 15] = i + 15

        self.P_out = jnp.zeros((21, 21), dtype=jnp.float32)
        for i in range(21):
            self.P_out = self.P_out.at[int(indices[i]), i].set(1.)

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs @ self.P_in)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean @ self.P_out - (unavail_actions * 1e10)
        greedy_action = jnp.argmax(action_logits, axis=-1)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return greedy_action, jnp.squeeze(critic, axis=-1)

class SymmetrizedActorCritic(nn.Module):
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

        return actor_mean, jnp.squeeze(critic, axis=-1)

class SymmActorCritic(nn.Module):
    action_dim: Sequence[int]
    config: Dict
    permutation_path: str

    def setup(self):
        self.layer_1_kernel, self.layer_1_bias, self.layer_2_kernel, self.layer_2_bias, self.layer_3_kernel, self.layer_3_bias, self.out_permutation = get_neural_symmetry(self.permutation_path)

    @nn.compact
    def __call__(self, x):
        obs, dones, avail_actions = x
        transformed_obs = nn.relu(obs @ self.layer_1_kernel + self.layer_1_bias)
        transformed_obs = nn.relu(transformed_obs @ self.layer_2_kernel + self.layer_2_bias)
        transformed_obs = nn.relu(transformed_obs @ self.layer_3_kernel + self.layer_3_bias)
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
        transformed_actor_mean = actor_mean @ self.out_permutation
        unavail_actions = 1 - avail_actions
        action_logits = transformed_actor_mean - (unavail_actions * 1e10)
        greedy_action = jnp.argmax(action_logits, axis=-1)

        critic = nn.Dense(512, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return greedy_action, jnp.squeeze(critic, axis=-1)

class TinIPPOAgent:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = load_params(weight_file)
        self.model = TinActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        greedy_action, value = self.model.apply(self.params, (obs, done, legal_moves))
        return greedy_action

class OblR2D2Agent(Agent):
    model: Any
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any]

    def act(self, obs: chex.Array, done, legal_actions: chex.Array, curr_player, rng, hidden_state) -> Tuple[chex.Array, Any]:
        obs = batchify(obs)
        legal_actions = batchify(legal_actions)
        priv_s = obs
        publ_s = obs[..., 125:]
        new_hidden_state, adv = self.apply_fn(self.params, hidden_state, (priv_s, publ_s))
        adv = adv.squeeze()
        legal_adv = (1 + adv - adv.min()) * legal_actions
        greedy_action = jnp.argmax(legal_adv, axis=-1)
        return greedy_action, new_hidden_state

    @classmethod
    def create(cls, model: OblR2D2, params):
        return cls(model=model, apply_fn=model.apply, params=params)

    def initialize_hidden_state(self):
        return self.model.initialize_carry(jax.random.PRNGKey(0), batch_dims=(2,))

class IPPOAgent:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)
        self.model = ActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng, hidden_state):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        greedy_action, value = self.model.apply(self.params, (obs, done, legal_moves))
        return greedy_action, hidden_state

class MDPSymmIPPOAgent:
    def __init__(self, weight_file, player_idx):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)
        self.model = MDPSymmActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng, hidden_state):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        greedy_action, value = self.model.apply(self.params, (obs, done, legal_moves))
        return greedy_action, hidden_state

class SymmIPPOAgent:
    def __init__(self, weight_file, player_idx, permutation_path):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)
        self.model = SymmActorCritic(env.action_space(env.agents[0]).n, config={}, permutation_path=permutation_path)

    def act(self, obs, done, legal_moves, curr_player, rng, hidden_state):
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        greedy_action, value = self.model.apply(self.params, (obs, done, legal_moves))
        return greedy_action, hidden_state

class MDPSymmetrizedIPPOAgent:
    def __init__(self, weight_file, player_idx, permutation_path="/workspace/models/hanabi/real_op_symmetries/"):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)

        self.num_permutations = 120

        self.in_permutations = jnp.zeros((120, 658, 658))
        self.out_permutations = jnp.zeros((120, 21, 21))
        for i in range(120):
            in_permutation, out_permutation = load_permutation(permutation_path, i)
            self.in_permutations = self.in_permutations.at[i, :].set(in_permutation)
            self.out_permutations = self.out_permutations.at[i, :].set(out_permutation)

        self.model = SymmetrizedActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng, hidden_state):
        # Batchify the inputs
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)
        
        def scan_fn(carry, idx):
            # Apply permutation and the model on each step
            in_permutation = self.in_permutations[idx]
            out_permutation = self.out_permutations[idx]
            
            # Transform the observation
            transformed_obs = obs @ in_permutation
            actor_mean, value = self.model.apply(self.params, (transformed_obs, done, legal_moves))
            
            # Update the accumulated actor_mean
            carry += actor_mean @ out_permutation
            
            return carry, None

        # Initialize carry for scan as zeros
        initial_actor_mean = jnp.zeros((2, 21))

        # Perform the scan over num_permutations
        final_actor_mean, _ = jax.lax.scan(scan_fn, initial_actor_mean, jnp.arange(self.num_permutations))

        # Compute the average across the accumulated actor_means
        actor_mean = final_actor_mean / self.num_permutations

        # Calculate unavailable actions mask
        unavail_actions = 1 - legal_moves

        # Apply the large negative mask to unavailable actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        # Create the distribution and sample an action
        # pi = distrax.Categorical(logits=action_logits)
        greedy_action = jnp.argmax(action_logits, axis=-1)
        return greedy_action, hidden_state


class SymmetrizedIPPOAgent:
    def __init__(self, weight_file, player_idx, permutation_paths):
        self.player_id = player_idx
        self.params = ippo_load_checkpoint(weight_file)

        self.num_permutations = len(permutation_paths.split(',')) + 1

        self.in_permutations_layer_1_kernel = jnp.zeros((self.num_permutations, 658, 658))
        self.in_permutations_layer_2_kernel = jnp.zeros((self.num_permutations, 658, 658))
        self.in_permutations_layer_3_kernel = jnp.zeros((self.num_permutations, 658, 658))
        self.in_permutations_layer_1_bias = jnp.zeros((self.num_permutations, 658))
        self.in_permutations_layer_2_bias = jnp.zeros((self.num_permutations, 658))
        self.in_permutations_layer_3_bias = jnp.zeros((self.num_permutations, 658))
        self.out_permutations = jnp.zeros((self.num_permutations, 21, 21))

        for i, permutation_path in enumerate(permutation_paths.split(',')):
            layer_1_kernel, layer_1_bias, layer_2_kernel, layer_2_bias, layer_3_kernel, layer_3_bias, out_permutation = get_neural_symmetry(permutation_path)

            self.in_permutations_layer_1_kernel = self.in_permutations_layer_1_kernel.at[i, :].set(layer_1_kernel)
            self.in_permutations_layer_2_kernel = self.in_permutations_layer_2_kernel.at[i, :].set(layer_2_kernel)
            self.in_permutations_layer_3_kernel = self.in_permutations_layer_3_kernel.at[i, :].set(layer_3_kernel)

            self.in_permutations_layer_1_bias = self.in_permutations_layer_1_bias.at[i, :].set(layer_1_bias)
            self.in_permutations_layer_2_bias = self.in_permutations_layer_2_bias.at[i, :].set(layer_2_bias)
            self.in_permutations_layer_3_bias = self.in_permutations_layer_3_bias.at[i, :].set(layer_3_bias)

            self.out_permutations = self.out_permutations.at[i, :].set(out_permutation)

        self.in_permutations_layer_1_kernel = self.in_permutations_layer_1_kernel.at[-1, :].set(jnp.eye(658))
        self.in_permutations_layer_2_kernel = self.in_permutations_layer_2_kernel.at[-1, :].set(jnp.eye(658))
        self.in_permutations_layer_3_kernel = self.in_permutations_layer_3_kernel.at[-1, :].set(jnp.eye(658))

        self.in_permutations_layer_1_bias = self.in_permutations_layer_1_bias.at[-1, :].set(jnp.zeros(658))
        self.in_permutations_layer_2_bias = self.in_permutations_layer_2_bias.at[-1, :].set(jnp.zeros(658))
        self.in_permutations_layer_3_bias = self.in_permutations_layer_3_bias.at[-1, :].set(jnp.zeros(658))

        self.out_permutations = self.out_permutations.at[-1, :].set(jnp.eye(21))

        self.model = SymmetrizedActorCritic(env.action_space(env.agents[0]).n, config={})

    def act(self, obs, done, legal_moves, curr_player, rng, hidden_state):
        # Batchify the inputs
        obs = batchify(obs)
        legal_moves = batchify(legal_moves)

        def scan_fn(carry, idx):
            # Access layer weights and biases for each permutation step
            in_perm_layer_1_kernel = self.in_permutations_layer_1_kernel[idx]
            in_perm_layer_1_bias = self.in_permutations_layer_1_bias[idx]
            in_perm_layer_2_kernel = self.in_permutations_layer_2_kernel[idx]
            in_perm_layer_2_bias = self.in_permutations_layer_2_bias[idx]
            in_perm_layer_3_kernel = self.in_permutations_layer_3_kernel[idx]
            in_perm_layer_3_bias = self.in_permutations_layer_3_bias[idx]
            out_permutation = self.out_permutations[idx]
            
            # Apply the three layers of transformations with ReLU activations
            transformed_obs = nn.relu(obs @ in_perm_layer_1_kernel + in_perm_layer_1_bias)
            transformed_obs = nn.relu(transformed_obs @ in_perm_layer_2_kernel + in_perm_layer_2_bias)
            transformed_obs = nn.relu(transformed_obs @ in_perm_layer_3_kernel + in_perm_layer_3_bias)

            # Get actor mean and value from the model
            actor_mean, value = self.model.apply(self.params, (transformed_obs, done, legal_moves))

            # Update the actor_mean accumulation
            carry += actor_mean @ out_permutation
            
            return carry, None

        # Initialize carry for scan as zeros
        initial_actor_mean = jnp.zeros((obs.shape[0], self.out_permutations[0].shape[1]))

        # Perform the scan over num_permutations
        final_actor_mean, _ = jax.lax.scan(scan_fn, initial_actor_mean, jnp.arange(self.num_permutations))

        # Compute the average across the accumulated actor_means
        actor_mean = final_actor_mean / self.num_permutations

        # Calculate unavailable actions mask
        unavail_actions = 1 - legal_moves

        # Apply the large negative mask to unavailable actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        # Create the distribution and sample an action
        # pi = distrax.Categorical(logits=action_logits)
        greedy_action = jnp.argmax(action_logits, axis=-1)
        return greedy_action, hidden_state

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
        elif player_type == "symm_ippo":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            agents.append(SymmIPPOAgent(weight_file, player_idx, (lambda i: args.symm0 if i == 0 else args.symm1)(player_idx)))
        elif player_type == "mdp_symm_ippo":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            agents.append(MDPSymmIPPOAgent(weight_file, player_idx))
        elif player_type == "symmetrized_ippo":
            agents.append(SymmetrizedIPPOAgent(weight_file, player_idx, (lambda i: args.symmetrized0 if i == 0 else args.symmetrized1)(player_idx)))
        elif player_type == "mdp_symmetrized_ippo":
            agents.append(MDPSymmetrizedIPPOAgent(weight_file, player_idx))
        elif player_type == "tin_ippo":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            agents.append(TinIPPOAgent(weight_file, player_idx))
        elif player_type == "obl":
            assert (
                weight_file is not None
            ), "Weight file must be provided for all the agents."
            model = OblR2D2()
            params = load_params(weight_file)
            obl_agent = OblR2D2Agent.create(model, params)
            agents.append(obl_agent)

    return agents

def play_game(rng, args):

    agents = get_agents(args)

    hidden_states = []
    for agent in agents:
        if hasattr(agent, 'initialize_hidden_state'):
            hidden_state = agent.initialize_hidden_state()
        else:
            hidden_state = None
        hidden_states.append(hidden_state)

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
            _, _, done, _, _, _, _, _ = val
            return jax.numpy.logical_not(done)

        def body_fn(val):
            cum_rew, t, done, rng, env_state, obs, legal_moves, hidden_states = val
            rng, _rng = jax.random.split(rng)
            curr_player = jnp.argmax(env_state.cur_player_idx)
            actions_all = []
            new_hidden_states = []
            for i in range(len(env.agents)):
                agent = agents[i]
                hidden_state = hidden_states[i]
                action, new_hidden_state = agent.act(obs, done, legal_moves, curr_player, _rng, hidden_state)
                actions_all.append(action)
                new_hidden_states.append(new_hidden_state)

            def true_fn(_):
                return {
                    agent: jnp.array(actions_all[1][i]) for i, agent in enumerate(env.agents)
                }

            def false_fn(_):
                return {
                    agent: jnp.array(actions_all[0][i]) for i, agent in enumerate(env.agents)
                }
            actions = jax.lax.cond(curr_player == 1, true_fn, false_fn, None)

            rng, env_state, obs, reward, dones, legal_moves = _step_env(
                rng, env_state, actions
            )

            done = dones["__all__"]
            cum_rew += reward["__all__"]
            t += 1
            return (cum_rew, t, done, rng, env_state, obs, legal_moves, new_hidden_states)

        init_val = (0, 0, False, rng, env_state, obs, legal_moves, hidden_states)
        cum_rew, _, _, _, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_val)

        return(cum_rew)

def play_games(rng_keys, args):
    play_game_vectorized = jax.vmap(play_game, in_axes=(0, None))
    results = play_game_vectorized(rng_keys, args)
    return f'{jnp.mean(results)} ± {jnp.std(results) / jnp.sqrt(len(results))}'

def main(args):
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0, 10000)

    rng = jax.random.PRNGKey(seed)
    rng_keys = jax.random.split(rng, args.num_rollouts)

    print(play_games(rng_keys, args))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player0", type=str, default="ippo")
    parser.add_argument("--player1", type=str, default="ippo")
    parser.add_argument("--symm0", type=str, default="")
    parser.add_argument("--symm1", type=str, default="")
    parser.add_argument("--symmetrized0", type=str, default="")
    parser.add_argument("--symmetrized1", type=str, default="")
    parser.add_argument("--weight0", type=str, default="PATH_TO_POLICY_1.pkl")
    parser.add_argument("--weight1", type=str, default="PATH_TO_POLICY_2.pkl")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_rollouts", type=int, default=5000)
    parser.add_argument("--use_jit", type=bool, default=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)