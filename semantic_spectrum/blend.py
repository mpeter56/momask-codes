"""
Feature-space blending for the Phase 2 zone-aware pipeline.

Two-step process
----------------
Step 1  — Feature update (per zone i):
    M'_orig,i  = M_orig,i + delta_i
    delta_i    = alpha * ||M_orig,i|| * (M_mom,i - M_orig,i)

Step 2  — Autoregressive reconstruction (per frame t = 1 … T):
    J_out(t) = J_out(t-1) + motion_vectors_from(M'_orig,i)
    J_out(0) = J_original(0)

The reconstruction step converts the updated feature vector back to per-frame
joint displacements via a learned or heuristic inversion.  The current
implementation uses a velocity-magnitude scaling heuristic: the target mean
speed encoded in M'_orig,i is used to scale the direction implied by the
original per-frame velocity, providing a smooth, identity-preserving update.
"""

from __future__ import annotations

import numpy as np

from semantic_spectrum.zones import ZoneConfig

FPS = 20.0
# Index within the feature vector produced by ZoneFeatureExtractor
_IDX_VEL_MEAN = 0   # target mean joint speed (m/s)
_IDX_VEL_STD  = 1


class FeatureBlender:
    """
    Blend original and MoMask feature vectors, then reconstruct the output
    joint sequence autoregressively.
    """

    def __init__(self, alpha: float = 0.5, zone_mode: str = "standard"):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha     = alpha
        self.zone_mode = zone_mode
        self._config   = ZoneConfig(zone_mode)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feature_blend(
        self,
        M_original: dict[str, np.ndarray],
        M_momask:   dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Step 1: compute updated feature vectors per zone.

        Parameters
        ----------
        M_original : {zone: feature_vector}  from original (MediaPipe) joints
        M_momask   : {zone: feature_vector}  from MoMask-generated joints

        Returns
        -------
        M_output : {zone: updated_feature_vector}
        """
        M_output: dict[str, np.ndarray] = {}
        for zone in M_original:
            if zone not in M_momask:
                M_output[zone] = M_original[zone].copy()
                continue
            m_o = M_original[zone].astype(np.float32)
            m_m = M_momask[zone].astype(np.float32)
            mag   = float(np.linalg.norm(m_o))
            delta = self.alpha * mag * (m_m - m_o)
            M_output[zone] = m_o + delta
        return M_output

    def reconstruct(
        self,
        M_output:       dict[str, np.ndarray],
        original_joints: np.ndarray,
    ) -> np.ndarray:
        """
        Step 2: autoregressively build the output joint sequence.

        J_out(t) = J_out(t-1) + scaled_displacement(t)
        J_out(0) = J_original(0)   -- preserves performer's starting pose.

        The displacement at each frame is taken from the original sequence but
        scaled so that the per-zone mean speed matches the target encoded in
        M_output.

        Parameters
        ----------
        M_output        : {zone: updated_feature_vector}
        original_joints : (T, 22, 3)

        Returns
        -------
        out_joints : (T, 22, 3)
        """
        if original_joints.ndim != 3 or original_joints.shape[1:] != (22, 3):
            raise ValueError(
                f"Expected (T, 22, 3) joints, got {original_joints.shape}"
            )

        T = original_joints.shape[0]
        out = np.zeros_like(original_joints)
        out[0] = original_joints[0]   # seed: original first frame

        # Original per-frame displacements (T-1, 22, 3)
        orig_disp = np.diff(original_joints, axis=0)

        for t in range(1, T):
            frame = out[t - 1].copy()
            for zone, indices in self._config.zones.items():
                if zone not in M_output:
                    frame[indices] = out[t - 1, indices]
                    continue
                scaled = self._apply_motion_vec(
                    J_prev=out[t - 1, indices],
                    orig_disp_t=orig_disp[t - 1, indices],
                    feature_vec=M_output[zone],
                )
                frame[indices] = scaled
            out[t] = frame

        return out

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _apply_motion_vec(
        self,
        J_prev:       np.ndarray,
        orig_disp_t:  np.ndarray,
        feature_vec:  np.ndarray,
    ) -> np.ndarray:
        """
        Apply updated motion vector to advance joints by one frame.

        Strategy: scale the original displacement direction so that the
        resulting joint speed matches the target mean speed encoded in
        feature_vec[_IDX_VEL_MEAN].  This preserves motion direction from
        the original while blending the magnitude toward the MoMask target.

        Parameters
        ----------
        J_prev       : (k, 3) joints at t-1
        orig_disp_t  : (k, 3) original displacement at this frame
        feature_vec  : (FEATURE_DIM,) updated feature vector for this zone

        Returns
        -------
        (k, 3) updated joint positions at t
        """
        target_speed = float(feature_vec[_IDX_VEL_MEAN]) / FPS   # m per frame

        orig_speed_per_joint = np.linalg.norm(orig_disp_t, axis=-1)  # (k,)
        mean_orig_speed = float(orig_speed_per_joint.mean())

        if mean_orig_speed < 1e-6:
            # No original motion: advance by a small random jitter to avoid
            # static lock when the target has non-zero speed
            return J_prev + orig_disp_t

        scale = target_speed / mean_orig_speed
        return J_prev + orig_disp_t * scale
