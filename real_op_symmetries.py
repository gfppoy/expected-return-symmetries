"""
Get Dec-POMDP symmetry transformations for 2-player Hanabi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import itertools
import pickle

perms = list(itertools.permutations([0,1,2,3,4]))


for perm_num, colour_map in enumerate(perms):
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

    P_in = jnp.zeros((658, 658), dtype=jnp.float32)
    for i in range(658):
        P_in = P_in.at[i, int(indices[i])].set(1.)

    indices = np.zeros(21)
    for i in range(10):
        indices[i] = i
    for i in range(5):
        indices[i + 10] = colour_map[i] + 10
    for i in range(6):
        indices[i + 15] = i + 15

    P_out = jnp.zeros((21, 21), dtype=jnp.float32)
    for i in range(21):
        P_out = P_out.at[int(indices[i]), i].set(1.)

    with open(f'/workspace/models/hanabi/real_op_symmetries/symmetry_{perm_num}_in.pkl', 'wb') as f:
        pickle.dump(P_in, f)

    with open(f'/workspace/models/hanabi/real_op_symmetries/symmetry_{perm_num}_out.pkl', 'wb') as f:
        pickle.dump(P_out, f)

    