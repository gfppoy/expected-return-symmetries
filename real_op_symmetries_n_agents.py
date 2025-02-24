"""
Get Dec-POMDP symmetry transformations for n-player Hanabi.
"""

import jax
import jax.numpy as jnp
import numpy as np
import itertools
import pickle

# Parameters
num_agents = 3
hand_size = 5
num_colors = 5
num_ranks = 5
deck_size = 50
max_info_tokens = 8
max_life_tokens = 3

# Calculate observation size for 3-player game
hands_n_feats = (num_agents - 1) * hand_size * num_colors * num_ranks + num_agents  # 250 + 3 = 253
board_n_feats = (deck_size - num_agents * hand_size) + num_colors * num_ranks + max_info_tokens + max_life_tokens  # 35 + 25 + 8 + 3 = 71
discards_n_feats = num_colors * 10  # 5 * 10 = 50
last_action_n_feats = (
        num_agents  # acting player index: 3
        + 4  # move type: 4
        + num_agents  # target player index: 3
        + num_colors  # color revealed: 5
        + num_ranks  # rank revealed: 5
        + hand_size  # reveal outcome
        + hand_size  # position played
        + num_colors * num_ranks  # card played/discarded: 25
        + 1  # card played/discarded: 1
        + 1  # card played score: 1
)  # Total: 57
v0_belief_n_feats = num_agents * hand_size * (num_colors * num_ranks + num_colors + num_ranks)  # 525

obs_size = hands_n_feats + board_n_feats + discards_n_feats + last_action_n_feats + v0_belief_n_feats  # 956
print(f"Computed observation size for 3-player game: {obs_size}")  # Should output 956

# Calculate number of moves for 3-player game
num_moves = 2 * hand_size + (num_agents - 1) * (num_colors + num_ranks) + 1  # 10 + 20 + 1 = 31
print(f"Number of moves for 3-player game: {num_moves}")  # Should output 31

# Action Encoding
agents = [f"agent_{i}" for i in range(num_agents)]
action_set = np.arange(num_moves)
action_encoding = {}
color_map = {0: 'Red', 1: 'Green', 2: 'Blue', 3: 'Yellow', 4: 'White'}  # Example color map


def _is_discard(a):
    return a < hand_size


def _is_play(a):
    return hand_size <= a < 2 * hand_size


def _is_hint_color(a):
    return 2 * hand_size <= a < 2 * hand_size + (num_agents - 1) * num_colors


def _is_hint_rank(a):
    return 2 * hand_size + (num_agents - 1) * num_colors <= a < 2 * hand_size + (num_agents - 1) * (num_colors + num_ranks)


for i, a in enumerate(action_set):
    if _is_discard(a):
        move_type = f'D{i % hand_size}'
    elif _is_play(a):
        move_type = f'P{i % hand_size}'
    elif _is_hint_color(a):
        action_idx = i - 2 * hand_size
        hint_idx = action_idx % num_colors
        target_player = action_idx // num_colors
        move_type = f'H{color_map[hint_idx]} to P{target_player + 1} relative'
    elif _is_hint_rank(a):
        action_idx = i - 2 * hand_size - (num_agents - 1) * num_colors
        hint_idx = action_idx % num_ranks
        target_player = action_idx // num_ranks
        move_type = f'H{hint_idx + 1} to P{target_player + 1} relative'
    else:
        move_type = 'N'
    action_encoding[i] = move_type

print(f"Action Encoding for 3-player game: {action_encoding}")

# Generate all color permutations
perms = list(itertools.permutations(range(num_colors)))

for perm_num, colour_map_perm in enumerate(perms):
    indices = np.zeros(obs_size, dtype=int)

    # 1. Hands Features
    # Assign hands for (num_agents - 1) other players
    for agent in range(num_agents - 1):
        for card in range(hand_size):
            for color in range(num_colors):
                for rank in range(num_ranks):
                    original_index = (
                            agent * hand_size * num_colors * num_ranks
                            + card * num_colors * num_ranks
                            + color * num_ranks
                            + rank
                    )
                    permuted_color = colour_map_perm[color]
                    permuted_index = (
                            agent * hand_size * num_colors * num_ranks
                            + card * num_colors * num_ranks
                            + permuted_color * num_ranks
                            + rank
                    )
                    indices[original_index] = permuted_index

    # 2. Missing Cards
    missing_cards_start = (num_agents - 1) * hand_size * num_colors * num_ranks  # 2 * 5 * 5 * 5 = 250
    for agent in range(num_agents):
        original_index = missing_cards_start + agent  # 250 + 0,1,2
        permuted_index = original_index  # Missing cards remain unchanged
        indices[original_index] = permuted_index

    # 3. Board Features
    board_start = missing_cards_start + num_agents  # 250 + 3 = 253
    for deck_feat in range(deck_size - num_agents * hand_size):  # 35 features
        original_index = board_start + deck_feat  # 253 to 287
        permuted_index = original_index  # Deck features remain unchanged
        indices[original_index] = permuted_index

    # Firework Features (permute colors)
    firework_start = board_start + (deck_size - num_agents * hand_size)  # 253 + 35 = 288
    for firework in range(num_colors):
        for rank in range(num_ranks):
            original_index = firework_start + firework * num_ranks + rank  # 288 + 0*5 +0 =288 to 288 +4*5+4=308
            permuted_firework = colour_map_perm[firework]
            permuted_index = firework_start + permuted_firework * num_ranks + rank
            indices[original_index] = permuted_index

    # Info Tokens and Life Tokens (not permuted)
    tokens_start = firework_start + num_colors * num_ranks  # 288 +25=313
    for token in range(max_info_tokens + max_life_tokens):  # 11 tokens
        original_index = tokens_start + token  # 313 to 323
        permuted_index = original_index  # Tokens remain unchanged
        indices[original_index] = permuted_index

    # 4. Discards Features (permute colors)
    discards_start = tokens_start + (max_info_tokens + max_life_tokens)  # 313 + 11 = 324
    for discard_color in range(num_colors):
        for rank in range(10):  # Assuming 10 possible discards per color
            original_index = discards_start + discard_color * 10 + rank  # 324 + 0*10 +0=324 to 324 +4*10+9=374
            permuted_color = colour_map_perm[discard_color]
            permuted_index = discards_start + permuted_color * 10 + rank
            indices[original_index] = permuted_index

    # 5. Last Action Features
    last_action_start = discards_start + num_colors * 10  # 324 +50=374

    # a. Acting Player Index (3 features) and other static features (additional 0 if necessary)
    for i in range(num_agents):  # 0-2
        original_index = last_action_start + i  # 374 +0=374 to 374 +2=376
        permuted_index = original_index
        indices[original_index] = permuted_index

    # b. Move Type (4 features)
    for i in range(4):  # 3-6
        original_index = last_action_start + num_agents + i  # 377 to 380
        permuted_index = original_index
        indices[original_index] = permuted_index

    # c. Target Player Index (3 features)
    for i in range(num_agents):  # 7-9
        original_index = last_action_start + num_agents + 4 + i  # 381 to 383
        permuted_index = original_index
        indices[original_index] = permuted_index

    # d. Color Revealed (5 features)
    for colour_revealed in range(num_colors):  # 10-14
        original_index = last_action_start + num_agents + 4 + num_agents + colour_revealed  # 384 to 388
        permuted_color = colour_map_perm[colour_revealed]
        permuted_index = last_action_start + num_agents + 4 + num_agents + permuted_color  # 384 + permuted_color
        indices[original_index] = permuted_index

    # e. Rank Revealed (5 features)
    for rank_revealed in range(num_ranks):  # 15-19
        original_index = last_action_start + num_agents + 4 + num_agents + num_colors + rank_revealed  # 389 to 393
        permuted_index = original_index  # Ranks are not permuted
        indices[original_index] = permuted_index

    # f. Hand Size (Rank Revealed) (5 features)
    for hs_rank_revealed in range(hand_size):  # 20-24
        original_index = last_action_start + num_agents + 4 + num_agents + num_colors + num_ranks + hs_rank_revealed  # 394 to 398
        permuted_index = original_index  # Positions are not permuted
        indices[original_index] = permuted_index

    # g. Hand Size (Reveal Outcome) (5 features)
    for hs_reveal_outcome in range(hand_size):  # 25-29
        original_index = last_action_start + num_agents + 4 + num_agents + num_colors + num_ranks + hand_size + hs_reveal_outcome  # 399 to 403
        permuted_index = original_index  # Positions are not permuted
        indices[original_index] = permuted_index

    # h. Position Played/Discarded (25 features)
    position_start = last_action_start + num_agents + 4 + num_agents + num_colors + num_ranks + 2 * hand_size  # 374 +3+4+3+5+5+10=374 +30=404
    for colour in range(num_colors):
        for rank in range(num_ranks):
            original_index = position_start + colour * num_ranks + rank  # 404 +0*5+0=404 to 404 +4*5+4=428
            permuted_color = colour_map_perm[colour]
            permuted_index = position_start + permuted_color * num_ranks + rank
            indices[original_index] = permuted_index

    # i. Card Played/Discarded (1 feature)
    card_played_discarded_index = last_action_start + num_agents + 4 + num_agents + num_colors + num_ranks + 2 * hand_size + num_colors * num_ranks  # 374 +30+25=429
    indices[card_played_discarded_index] = card_played_discarded_index  # Remains unchanged

    # j. Card Played Score (1 feature)
    card_played_score_index = card_played_discarded_index + 1  # 430
    indices[card_played_score_index] = card_played_score_index  # Remains unchanged

    # Total features assigned in last_action: from 374 to 430 inclusive =57

    # 6. V0 Belief Features
    v0_belief_start = card_played_score_index + 1  # 431
    curr_idx = v0_belief_start
    for card in range(num_agents * hand_size):  # 15 cards
        # a. Card Features (Color and Rank)
        for colour in range(num_colors):
            for rank in range(num_ranks):
                original_index = curr_idx + colour * num_ranks + rank  # 431 +0*5+0=431 to 431 +4*5+4=455
                permuted_color = colour_map_perm[colour]
                permuted_index = curr_idx + permuted_color * num_ranks + rank
                indices[original_index] = permuted_index
        curr_idx += num_colors * num_ranks  # +25 => 431 +25=456

        # b. Revealed Colors (permute colors)
        for colour in range(num_colors):
            original_index = curr_idx + colour  # 456 +0=456 to 456 +4=460
            permuted_color = colour_map_perm[colour]
            permuted_index = curr_idx + permuted_color
            indices[original_index] = permuted_index
        curr_idx += num_colors  # +5 => 456 +5=461

        # c. Revealed Ranks (not permuted)
        for rank in range(num_ranks):
            original_index = curr_idx + rank  # 461 +0=461 to 461 +4=465
            permuted_index = original_index  # Ranks are not permuted
            indices[original_index] = permuted_index
        curr_idx += num_ranks  # +5 => 461 +5=466

    # 7. Creating P_in Matrix
    P_in = jnp.zeros((obs_size, obs_size), dtype=jnp.float32)
    for i in range(obs_size):
        P_in = P_in.at[i, int(indices[i])].set(1.)

    # 8. Creating P_out Mapping for 3-player game
    # For 3p, num_moves =31
    output_size = num_moves  # 31
    indices_out = np.zeros(output_size, dtype=int)

    # 1. Discard and Play Actions (0-9): No permutation
    for i in range(2 * hand_size):  # 0-9
        indices_out[i] = i  # These actions remain unchanged

    # 2. Hint Color to Player 1 (10-14) and Player 2 (15-19): Permute colors
    for i in range(num_colors):  # 0-4
        # Hint Color to Player 1
        original_action_p1 = 2 * hand_size + i  # 10 + i
        permuted_color = colour_map_perm[i]
        permuted_action_p1 = 2 * hand_size + permuted_color  # 10 + permuted_color
        indices_out[original_action_p1] = permuted_action_p1

        # Hint Color to Player 2
        original_action_p2 = 2 * hand_size + num_colors + i  # 15 + i
        permuted_action_p2 = 2 * hand_size + num_colors + permuted_color  # 15 + permuted_color
        indices_out[original_action_p2] = permuted_action_p2

    # 3. Hint Rank to Player 1 (20-24) and Player 2 (25-29): No permutation
    for i in range(num_ranks):  # 0-4
        # Hint Rank to Player 1
        original_action_hr_p1 = 2 * hand_size + (num_agents - 1) * num_colors + i  # 20 + i
        indices_out[original_action_hr_p1] = original_action_hr_p1  # No change

        # Hint Rank to Player 2
        original_action_hr_p2 = 2 * hand_size + (num_agents - 1) * num_colors + num_ranks + i  # 25 + i
        indices_out[original_action_hr_p2] = original_action_hr_p2  # No change

    # 4. Noop Action (30): No permutation
    noop_action = num_moves - 1  # 30
    indices_out[noop_action] = noop_action  # Remains unchanged

    # Verify that all indices_out are within bounds
    assert np.all(indices_out < output_size), f"Some action indices are out of bounds: {indices_out[indices_out >= output_size]}"

    # Create P_out matrix
    P_out = jnp.zeros((output_size, output_size), dtype=jnp.float32)
    for i in range(output_size):
        P_out = P_out.at[int(indices_out[i]), i].set(1.)

    # 9. Saving the Symmetry Mappings
    with open(f'/workspace/models/hanabi/real_op_symmetries_3p/symmetry_{perm_num}_in_3p.pkl', 'wb') as f:
        pickle.dump(P_in, f)

    with open(f'/workspace/models/hanabi/real_op_symmetries_3p/symmetry_{perm_num}_out_3p.pkl', 'wb') as f:
        pickle.dump(P_out, f)

    print(f"Saved symmetry {perm_num} for 3-player game.")