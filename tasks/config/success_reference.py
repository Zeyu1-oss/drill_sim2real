"""
Per-variant success grasp reference templates.
Derived from SUCCESS logs: wrist position + quaternion + proximal finger joint angles
when the policy successfully grasps each drill variant.

Per-dimension statistics based on n=100 samples per variant.
Arrays are indexed as: [index, middle, pinky, ring, thumb_yaw, thumb_pitch] for joints,
[x, y, z] for wrist_pos.
"""

import numpy as np

SUCCESS_REFERENCE = {
    0: {
        "name": "drill2",
        "joint_mean": np.array([0.8994, 1.1452, 1.4612, 1.3506, 1.2612, 0.0700]),
        "joint_threshold": np.array([0.0727, 0.2113, 0.0412, 0.0657, 0.0725, 0.0722]),
        "wrist_pos_mean": np.array([-0.0457, 0.0236, -0.0971]),
        "wrist_pos_threshold": np.array([0.0149, 0.0173, 0.0109]),
    },
    1: {
        "name": "drill_blue",
        "joint_mean": np.array([0.7786, 1.0400, 1.2686, 1.1765, 1.2528, -0.0004]),
        "joint_threshold": np.array([0.0490, 0.0705, 0.0660, 0.0312, 0.0618, 0.002]),
        "wrist_pos_mean": np.array([0.0907, 0.0692, -0.0378]),
        "wrist_pos_threshold": np.array([0.0188, 0.0211, 0.008]),
    },
    2: {
        "name": "drill_yellow",
        "joint_mean": np.array([0.7216, 0.8495, 1.3452, 1.1865, 1.0513, -0.0002]),
        "joint_threshold": np.array([0.0644, 0.1275, 0.0284, 0.1388, 0.1457, 0.002]),
        "wrist_pos_mean": np.array([0.1103, 0.0542, 0.1284]),
        "wrist_pos_threshold": np.array([0.0123, 0.0176, 0.0142]),
    },
}
