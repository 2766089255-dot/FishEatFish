# utils.py
# Utility functions for bot level distribution

import numpy as np
from config import MIN_LEVEL, MAX_LEVEL, LEVEL_DIST_SIGMA

def truncated_normal_prob(center, sigma=LEVEL_DIST_SIGMA):
    """
    Generate a truncated normal probability distribution over bot levels.
    The distribution is centered at `center` (usually AI level + 1) and
    clipped to the valid range [MIN_LEVEL, MAX_LEVEL-1].
    Returns a list of probabilities for levels 1..MAX_LEVEL-1.
    """
    levels = np.arange(MIN_LEVEL, MAX_LEVEL)          # 1 to MAX_LEVEL-1
    weights = np.exp(-0.5 * ((levels - center) / sigma) ** 2)
    if weights.sum() == 0:
        probs = np.ones(len(levels)) / len(levels)    # fallback to uniform
    else:
        probs = weights / weights.sum()
    return probs.tolist()

def sample_level(prob_list):
    """
    Sample a bot level from a probability list.
    """
    levels = np.arange(MIN_LEVEL, MAX_LEVEL)
    return np.random.choice(levels, p=prob_list)