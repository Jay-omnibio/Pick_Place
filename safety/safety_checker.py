"""
Safety checker: validates/modifies actions before application.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class SafetyChecker:
    """
    Checks and optionally clamps actions for safety.
    - Workspace bounds
    - Max velocity (move delta magnitude)
    - Optional episode timeout (caller tracks steps)
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.workspace_min = np.array(
            config.get("workspace_min", [0.05, -0.50, 0.005]), dtype=float
        )
        self.workspace_max = np.array(
            config.get("workspace_max", [0.85, 0.50, 0.95]), dtype=float
        )
        self.max_move_norm = float(config.get("max_move_norm", 0.05))
        self.max_ee_target_radius = float(config.get("max_ee_target_radius", 0.85))

    def check(
        self,
        action: Dict,
        current_ee_pos: Optional[np.ndarray] = None,
    ) -> Optional[Dict]:
        """
        Validate/clamp action. Returns modified action or None to abort.
        current_ee_pos: used when action has ee_target_pos to clamp target.
        """
        if action is None:
            return None
        action = dict(action)

        if "move" in action and action.get("move"):
            move = np.asarray(action["move"], dtype=float).reshape(3)
            n = np.linalg.norm(move)
            if n > self.max_move_norm and n > 0:
                move = (move / n) * self.max_move_norm
            action["move"] = move.tolist()

        if "ee_target_pos" in action and current_ee_pos is not None:
            target = np.asarray(action["ee_target_pos"], dtype=float).reshape(3)
            target = np.clip(target, self.workspace_min, self.workspace_max)
            radial = np.linalg.norm(target[:2])
            if radial > self.max_ee_target_radius and radial > 0:
                target[:2] = target[:2] / radial * self.max_ee_target_radius
            action["ee_target_pos"] = target.tolist()

        return action
