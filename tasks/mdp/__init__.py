"""
MDP (Markov Decision Process) 相关函数
包含奖励、终止条件、观察等函数
"""

# 只导入终止条件（不需要 rewards.py）
from .terminations import *

__all__ = [
    # 终止条件
    "check_success",
    "check_failure",
]
