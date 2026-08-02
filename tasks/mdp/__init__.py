"""
MDP (Markov Decision Process) functions.
Rewards, terminations, observations, etc.
"""

# Only import terminations (rewards.py not needed).
from .terminations import *

__all__ = [
    # terminations
    "check_success",
    "check_failure",
]
