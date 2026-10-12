"""Decode stored PRIMEU absolute targets, retaining unpredicted starting joints.

Chebyshev evaluation is the pure decoding part of the dataset generation tools
under primeu_data/_hand_reduction_tools/*/hand_curves.py. No fitting, kinematics,
SciPy, hardware access, relative-action transform, or basis replacement occurs.
"""
from pathlib import Path

import numpy as np
from numpy.polynomial.chebyshev import chebvander

from .contract import STATE_SLICES, load_checkpoint_contract, verify_basis_files


def decode_finger(coordinates, curve):
    """Match the source decoder's coordinate clipping and optional joint limits."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("Finger coordinates must be finite")
    coefficients = np.asarray(curve["coefficients"], dtype=np.float64)
    result = chebvander(np.clip(coordinates, -1.0, 1.0), len(coefficients) - 1) @ coefficients
    if curve.get("clip_joint_limits", False):
        result = np.clip(result, curve["joint_lower"], curve["joint_upper"])
    return result


class ActionCodec:
    def __init__(self, contract, basis_dir):
        self.contract = contract
        self.bases = verify_basis_files(contract, basis_dir)
        self._held_state = None

    @classmethod
    def from_directory(cls, checkpoint):
        return cls(load_checkpoint_contract(checkpoint), Path(checkpoint) / "primeu")

    def reset(self):
        """Forget starting posture only; this never calls a robot or moves home."""
        self._held_state = None

    def decode(self, actions, state):
        actions = np.asarray(actions, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != self.contract.action_dim:
            raise ValueError(f"Expected nonempty [H,{self.contract.action_dim}] action chunk")
        if state.shape != (60,) or not np.isfinite(state).all() or not np.isfinite(actions).all():
            raise ValueError("Expected finite action chunk and finite 60D physical state")
        held = state if self._held_state is None else self._held_state
        output = np.broadcast_to(held, (len(actions), 60)).copy()
        for group, bounds in self.contract.action_slices.items():
            values = actions[:, bounds["start"]:bounds["end"]]
            target_group = group.replace("_synergy", "")
            if group == "left_hand_synergy":
                values = np.concatenate([values[:, :4]] + [
                    decode_finger(values[:, 4 + i], curve)
                    for i, curve in enumerate(self.bases["left"]["fingers"])], axis=-1)
            elif group == "right_hand_synergy":
                values = np.concatenate([
                    decode_finger(values[:, i], curve)
                    for i, curve in enumerate(self.bases["right"]["fingers"])], axis=-1)
            target = STATE_SLICES[target_group]
            output[:, target["start"]:target["end"]] = values
        if not np.isfinite(output).all():
            raise ValueError("Decoded PRIMEU actions are non-finite")
        # Invalid calls never latch a partially validated start pose.
        if self._held_state is None:
            self._held_state = state.copy()
        return output

    expand = decode
