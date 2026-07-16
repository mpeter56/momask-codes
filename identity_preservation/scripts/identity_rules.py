"""
identity_rules.py — 10 source-only perturbation rules.

Redesigned after studying github.com/soheiw/unnoticed_dance.

Key principle: NEVER generate new motion. Only perturb the source. Every
rule takes only the source pose sequence and a strength; the output is a
time-warped and/or spatially nudged version of the same person doing the
same recording. Identity is preserved by construction.

Signature:
    fn(source: (T, 22, 3), strength: float, fps: int = 20) -> (T, 22, 3)

The 10 rules:
    Time-based (5)
      1. WAVE          — sinusoidal time-shift
      2. UPPER PULL    — time-shift with upper-body bias
      3. CENTER RIPPLE — time-shift with torso bias
      4. FLOAT DRIFT   — slow cosine time-shift
      5. GESTURE ACCENT— nonlinear time-warp toward motion peaks
      6. RHYTHM LOCK   — snap-to-BPM time-warp

    Space-based (4)
      7. SPINE SWAY    — micro-yaw around vertical axis
      8. BREATH SCALE  — bone-length oscillation with breathing rhythm
      9. HEAD HOLD     — head + hands damped toward their mean position
     10. ROOT FREEZE   — pelvis damped toward mean, limbs keep their motion
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ------------------------------------------------------------------
# SMPL-H 22-joint indexing (HumanML3D convention)
# ------------------------------------------------------------------
# 0=pelvis, 1=L hip, 2=R hip, 3=spine, 4=L knee, 5=R knee, 6=spine1,
# 7=L ankle, 8=R ankle, 9=spine2/chest, 10=L toe, 11=R toe, 12=neck,
# 13=L shoulder, 14=R shoulder, 15=head, 16=L elbow, 17=R elbow,
# 18=L wrist, 19=R wrist, 20=L hand-tip, 21=R hand-tip
PARENTS      = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
UPPER_JOINTS = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_JOINTS = [0, 1, 2, 4, 5, 7, 8, 10, 11]
FEET_JOINTS  = [7, 8, 10, 11]
FACE_ANCHOR  = [12, 15]                 # neck + head (face never moves relative to itself)
HAND_ANCHOR  = [18, 19, 20, 21]         # wrists + hand tips


def _anchor_weights(strength: float = 1.0) -> np.ndarray:
    """Weights that fully anchor face + hands, apply the shift fully everywhere else.
    (Face + hands = the strongest identity cues — never let them time-warp.)"""
    w = np.ones(22, dtype=np.float32) * strength
    for j in FACE_ANCHOR + HAND_ANCHOR:
        w[j] = 0.0
    return w


def apply_time_shift(source: np.ndarray, shift_per_frame_ms: np.ndarray,
                     joint_weights: np.ndarray | None = None, fps: int = 20) -> np.ndarray:
    """
    Vectorised per-joint per-frame time-shift with linear interpolation.

    source              : (T, 22, 3)
    shift_per_frame_ms  : (T,) — signed ms offset from the current playhead
    joint_weights       : (22,) — 0 = frozen at original time, 1 = full shift
    """
    T = source.shape[0]
    frame_ms = 1000.0 / fps
    if joint_weights is None:
        joint_weights = _anchor_weights()

    # (T, 22) — per-frame per-joint shift in frames
    shift_frames = (shift_per_frame_ms[:, None] * joint_weights[None, :]) / frame_ms
    frame_idx = np.arange(T)[:, None] + shift_frames                # (T, 22)
    frame_idx = np.clip(frame_idx, 0, T - 1)

    i0 = np.floor(frame_idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, T - 1)
    alpha = (frame_idx - i0)[..., None].astype(np.float32)          # (T, 22, 1)

    j_idx = np.broadcast_to(np.arange(22)[None, :], (T, 22))        # (T, 22)
    p0 = source[i0, j_idx]                                          # (T, 22, 3)
    p1 = source[i1, j_idx]
    return ((1.0 - alpha) * p0 + alpha * p1).astype(np.float32)


# ==================================================================
# Time-based rules
# ==================================================================
def rule_wave(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R1 — WAVE: sinusoidal time-shift, ±10 ms amplitude at strength=1."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    shift_ms = (np.sin(ms / 500.0) * 10.0 + np.cos(ms / 700.0) * 8.0) * strength
    return apply_time_shift(source, shift_ms.astype(np.float32), _anchor_weights(), fps=fps)


def rule_upper_pull(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R2 — UPPER PULL: time-shift biased to upper body (torso + arms + head).
    Lower body plays back on the true timeline. Legs walk, torso wobbles."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    shift_ms = (np.sin(ms / 680.0) * 18.0 * strength).astype(np.float32)
    w = _anchor_weights()
    for j in LOWER_JOINTS:
        w[j] *= 0.15                    # lower body barely shifts
    return apply_time_shift(source, shift_ms, w, fps=fps)


def rule_center_ripple(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R3 — CENTER RIPPLE: time-shift concentrated on torso, decays at extremities."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    shift_ms = (np.sin(ms / 360.0) * 32.0 * strength).astype(np.float32)
    w = _anchor_weights()
    for j in FEET_JOINTS:
        w[j] *= 0.25
    for j in (18, 19):                  # wrists — a bit damped
        w[j] *= 0.4
    return apply_time_shift(source, shift_ms, w, fps=fps)


def rule_float_drift(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R4 — FLOAT DRIFT: slow cosine time-shift, long-period wobble."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    shift_ms = (np.cos(ms / 820.0) * 24.0 * strength).astype(np.float32)
    return apply_time_shift(source, shift_ms, _anchor_weights(), fps=fps)


def rule_gesture_accent(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R5 — GESTURE ACCENT: nonlinear time-warp that lingers near motion peaks
    (moments where the source is moving fastest) and skips through the quiet
    moments — accentuates gestures already in the recording."""
    T = source.shape[0]
    # per-frame kinetic energy of the source
    vel = np.linalg.norm(np.diff(source, axis=0), axis=(-1, -2))     # (T-1,)
    vel = np.concatenate([[vel[0]], vel]).astype(np.float32)          # (T,)
    vel_norm = (vel - vel.mean()) / (vel.std() + 1e-9)
    # Positive shift where motion is above average (linger), negative where quiet (skip)
    shift_ms = (vel_norm * 40.0 * strength).astype(np.float32)
    return apply_time_shift(source, shift_ms, _anchor_weights(), fps=fps)


def rule_rhythm_lock(source: np.ndarray, strength: float = 1.0, fps: int = 20,
                     bpm: float = 120.0) -> np.ndarray:
    """R6 — RHYTHM LOCK: snap playhead time toward the nearest 120-BPM beat."""
    T = source.shape[0]
    beat_ms = 60000.0 / bpm
    ms = np.arange(T) * (1000.0 / fps)
    # signed distance to nearest beat, in [-beat/2, beat/2]
    phase = ((ms + beat_ms / 2.0) % beat_ms) - beat_ms / 2.0
    shift_ms = (-phase * 0.35 * strength).astype(np.float32)         # pull toward beat
    return apply_time_shift(source, shift_ms, _anchor_weights(), fps=fps)


# ==================================================================
# Space-based rules — all still operating on the source only
# ==================================================================
def rule_spine_sway(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R7 — SPINE SWAY: micro-rotation around the vertical (Y) axis, ±5° max.
    Pivot is the pelvis so the rest of the pose sways around it."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    theta = np.radians(np.sin(ms / 600.0) * 5.0 * strength).astype(np.float32)  # (T,)
    cos_t = np.cos(theta)[:, None, None]
    sin_t = np.sin(theta)[:, None, None]

    root = source[:, 0:1, :]
    centred = source - root
    x = centred[..., 0:1]; y = centred[..., 1:2]; z = centred[..., 2:3]
    x_new =  cos_t * x + sin_t * z
    z_new = -sin_t * x + cos_t * z
    rotated = np.concatenate([x_new, y, z_new], axis=-1)
    return (rotated + root).astype(np.float32)


def rule_breath_scale(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R8 — BREATH SCALE: subtle bone-length oscillation at ~15 breaths/min,
    max ±2% at full strength. The body pulses as if breathing."""
    T = source.shape[0]
    ms = np.arange(T) * (1000.0 / fps)
    breath_hz = 0.25
    scale = 1.0 + np.sin(ms / 1000.0 * 2 * np.pi * breath_hz) * 0.02 * strength   # (T,)
    root = source[:, 0:1, :]
    return (root + (source - root) * scale[:, None, None]).astype(np.float32)


def rule_head_hold(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R9 — HEAD HOLD: pull the head + hands damped toward their mean position.
    Emphasises 'the identity's face + hands are always here' — strong identity
    anchor, mild pose modification of the extremities themselves."""
    result = source.copy()
    for j in FACE_ANCHOR + HAND_ANCHOR:
        mean_pos = source[:, j:j+1, :].mean(axis=0, keepdims=True)      # (1, 1, 3)
        # blend the joint's motion toward its mean — 30% pull at strength=1
        result[:, j:j+1, :] = source[:, j:j+1, :] * (1.0 - strength * 0.3) \
                              + mean_pos * (strength * 0.3)
    return result.astype(np.float32)


def rule_root_freeze(source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    """R10 — ROOT FREEZE: pull the pelvis position toward its mean (freezing the
    walking trajectory) — limbs still articulate but the identity stays in one
    spot. Great for keeping the figure centred on stage."""
    root = source[:, 0:1, :]
    root_mean = root.mean(axis=0, keepdims=True)                        # (1, 1, 3)
    new_root = root * (1.0 - strength) + root_mean * strength
    offset = new_root - root                                            # (T, 1, 3)
    return (source + offset).astype(np.float32)


# ==================================================================
# Registry + metrics
# ==================================================================
@dataclass
class Rule:
    key: str
    title: str
    tagline: str
    preserves: str
    borrows: str
    fn: Callable[..., np.ndarray]


RULES: list[Rule] = [
    Rule("wave",           "WAVE",           "sinusoidal time-shift",              "identity (face + hands frozen)", "±10 ms body-wide playhead wobble",   rule_wave),
    Rule("upper_pull",     "UPPER PULL",     "upper-body time-shift",              "leg motion + face + hands",      "±18 ms wobble in torso + arms",     rule_upper_pull),
    Rule("center_ripple",  "CENTER RIPPLE",  "torso-centred time-shift",           "feet + hands + face",            "±32 ms wobble in spine + hips",     rule_center_ripple),
    Rule("float_drift",    "FLOAT DRIFT",    "slow cosine time-shift",             "identity (face + hands frozen)", "±24 ms slow global drift",          rule_float_drift),
    Rule("gesture_accent", "GESTURE ACCENT", "linger on motion peaks",             "identity (face + hands frozen)", "time re-mapped to accent gestures", rule_gesture_accent),
    Rule("rhythm_lock",    "RHYTHM LOCK",    "snap to 120-BPM grid",               "identity (face + hands frozen)", "time snapped to beats",             rule_rhythm_lock),
    Rule("spine_sway",     "SPINE SWAY",     "±5° yaw around pelvis",              "pose articulation",              "small rotational sway",             rule_spine_sway),
    Rule("breath_scale",   "BREATH SCALE",   "±2% breathing pulse",                "pose + trajectory",              "expand/contract at breath rate",    rule_breath_scale),
    Rule("head_hold",      "HEAD HOLD",      "head + hands damped to their mean",  "body pose",                      "head + hands anchored to mean",     rule_head_hold),
    Rule("root_freeze",    "ROOT FREEZE",    "pelvis damped to its mean",          "limb articulation",              "pelvis trajectory pulled to centre", rule_root_freeze),
]


def apply_rule(key: str, source: np.ndarray, strength: float = 1.0, fps: int = 20) -> np.ndarray:
    for r in RULES:
        if r.key == key:
            return r.fn(source, strength=strength, fps=fps).astype(np.float32)
    raise KeyError(f"unknown rule '{key}' — valid: {[r.key for r in RULES]}")


def perturbation_amount(source: np.ndarray, blended: np.ndarray) -> float:
    """Mean per-frame per-joint L2 displacement, normalised by pose scale.
    Values in roughly [0, 0.3]. Higher = more visible dance, identity still
    preserved by construction because 'blended' is derived from 'source'."""
    diff = np.linalg.norm(blended - source, axis=-1).mean()               # scalar
    scale = np.linalg.norm(source, axis=-1).mean() + 1e-9
    return float(diff / scale)
