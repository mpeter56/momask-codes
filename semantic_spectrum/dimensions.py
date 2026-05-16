"""
Semantic dimension definitions with kinematic feature extractors.

Each dimension maps to specific HumanML3D features (263-dim vector):
  [0]       root_rot_velocity
  [1:3]     root_linear_velocity (x, z)
  [3]       root_y
  [4:67]    ric_data  (joint_num-1)*3 = 21*3 = 63
  [67:193]  rot_data  (joint_num-1)*6 = 21*6 = 126
  [193:256] local_velocity  joint_num*3 = 22*3 = 66
  [256:260] foot_contact (4 binary)
"""

import numpy as np

# HumanML3D joint indices (22 joints)
# 0:Pelvis 1:LHip 2:RHip 3:Spine1 4:LKnee 5:RKnee 6:Spine2
# 7:LAnkle 8:RAnkle 9:Spine3 10:LFoot 11:RFoot 12:Neck
# 13:LCollar 14:RCollar 15:Head 16:LShoulder 17:RShoulder
# 18:LElbow 19:RElbow 20:LWrist 21:RWrist

# RIC data: joint_id -> feature slice [4 + joint_id*3 : 4 + joint_id*3 + 3]
# Rot data: joint_id -> feature slice [67 + joint_id*6 : 67 + joint_id*6 + 6]
# Local vel: joint_id -> feature slice [193 + joint_id*3 : 193 + joint_id*3 + 3]

UPPER_BODY_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_BODY_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
ARM_JOINTS = [16, 17, 18, 19, 20, 21]
LEG_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]


def _ric_slice(joint_id):
    return slice(4 + joint_id * 3, 4 + joint_id * 3 + 3)


def _rot_slice(joint_id):
    return slice(67 + joint_id * 6, 67 + joint_id * 6 + 6)


def _vel_slice(joint_id):
    return slice(193 + joint_id * 3, 193 + joint_id * 3 + 3)


def extract_walk(motion: np.ndarray) -> float:
    """Steady forward velocity with foot contact alternation and low root-y variance."""
    root_vel = motion[:, 1:3]           # (T, 2) forward + lateral
    forward_speed = np.mean(np.abs(root_vel[:, 0]))
    lateral_speed = np.mean(np.abs(root_vel[:, 1]))
    directionality = forward_speed / (lateral_speed + 1e-6)
    foot = motion[:, 256:260]           # (T, 4)
    # Alternating foot contact: variance across time is higher for walking than standing
    foot_alt = np.std(foot[:, 0] - foot[:, 2]) + np.std(foot[:, 1] - foot[:, 3])
    root_y_var = np.var(motion[:, 3])
    # Combine: high forward speed, directional, alternating feet, low vertical bounce
    score = (
        np.clip(forward_speed / 2.0, 0, 1) * 0.35
        + np.clip(directionality / 5.0, 0, 1) * 0.25
        + np.clip(foot_alt / 1.5, 0, 1) * 0.25
        + np.clip(1.0 - root_y_var * 10, 0, 1) * 0.15
    )
    return float(score)


def extract_dance(motion: np.ndarray) -> float:
    """Rhythmic articulation, expressive upper body, high local joint velocity variance."""
    upper_vel = np.concatenate(
        [motion[:, _vel_slice(j)] for j in ARM_JOINTS], axis=1
    )
    upper_energy = np.mean(np.linalg.norm(upper_vel, axis=1))
    rot_var = np.mean([np.var(motion[:, _rot_slice(j)]) for j in UPPER_BODY_JOINTS])
    root_rot_var = np.var(motion[:, 0])  # root rotation velocity
    score = (
        np.clip(upper_energy / 3.0, 0, 1) * 0.40
        + np.clip(rot_var * 20, 0, 1) * 0.35
        + np.clip(root_rot_var * 5, 0, 1) * 0.25
    )
    return float(score)


def extract_run(motion: np.ndarray) -> float:
    """High forward speed, both feet off ground (airborne phases)."""
    forward_speed = np.mean(np.abs(motion[:, 1]))
    foot = motion[:, 256:260]
    airborne = np.mean((foot[:, 0] + foot[:, 2]) < 0.5)   # both feet up
    score = (
        np.clip(forward_speed / 4.0, 0, 1) * 0.60
        + np.clip(airborne / 0.3, 0, 1) * 0.40
    )
    return float(score)


def extract_jump(motion: np.ndarray) -> float:
    """High root-y peaks and airborne phases."""
    root_y = motion[:, 3]
    y_range = root_y.max() - root_y.min()
    foot = motion[:, 256:260]
    airborne = np.mean((foot[:, 0] + foot[:, 2]) < 0.5)
    score = (
        np.clip(y_range / 0.5, 0, 1) * 0.50
        + np.clip(airborne / 0.4, 0, 1) * 0.50
    )
    return float(score)


def extract_spin(motion: np.ndarray) -> float:
    """High root rotation velocity sustained over time."""
    root_rot = np.abs(motion[:, 0])
    sustained = np.percentile(root_rot, 75)
    score = np.clip(sustained / 2.0, 0, 1)
    return float(score)


def extract_kick(motion: np.ndarray) -> float:
    """High leg velocity with single foot loss of contact."""
    leg_vel = np.concatenate(
        [motion[:, _vel_slice(j)] for j in LEG_JOINTS], axis=1
    )
    leg_energy = np.max(np.linalg.norm(leg_vel, axis=1))
    foot = motion[:, 256:260]
    one_foot_up = np.mean(
        ((foot[:, 0] < 0.5) & (foot[:, 2] > 0.5)) |
        ((foot[:, 0] > 0.5) & (foot[:, 2] < 0.5))
    )
    score = (
        np.clip(leg_energy / 5.0, 0, 1) * 0.55
        + np.clip(one_foot_up / 0.5, 0, 1) * 0.45
    )
    return float(score)


def extract_wave(motion: np.ndarray) -> float:
    """Oscillatory arm motion while body stays relatively still."""
    arm_vel = np.concatenate(
        [motion[:, _vel_slice(j)] for j in [20, 21]], axis=1   # wrists
    )
    arm_energy = np.mean(np.linalg.norm(arm_vel, axis=1))
    root_vel = np.mean(np.abs(motion[:, 1:3]))
    score = (
        np.clip(arm_energy / 2.0, 0, 1) * 0.60
        + np.clip(1.0 - root_vel * 3, 0, 1) * 0.40
    )
    return float(score)


def extract_stand(motion: np.ndarray) -> float:
    """Minimal overall motion — low velocity everywhere, feet on ground."""
    all_vel = motion[:, 193:256]
    total_energy = np.mean(np.linalg.norm(all_vel, axis=1))
    foot = motion[:, 256:260]
    grounded = np.mean(foot[:, 0] + foot[:, 2])
    score = (
        np.clip(1.0 - total_energy / 1.0, 0, 1) * 0.60
        + np.clip(grounded / 2.0, 0, 1) * 0.40
    )
    return float(score)


# Registry: name -> (extractor_fn, natural_language_descriptor)
DIMENSIONS = {
    "walk":  (extract_walk,  "walks forward"),
    "dance": (extract_dance, "dances with rhythmic movement"),
    "run":   (extract_run,   "runs"),
    "jump":  (extract_jump,  "jumps"),
    "spin":  (extract_spin,  "spins"),
    "kick":  (extract_kick,  "kicks"),
    "wave":  (extract_wave,  "waves"),
    "stand": (extract_stand, "stands"),
}
