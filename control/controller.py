import numpy as np


class EEController:
    """
    End-effector level controller.

    Converts abstract actions into low-level commands.
    """

    def __init__(self, simulator, config=None):
        self.simulator = simulator

        # Safety limits
        self.max_step = 0.05   # meters per step
        self.min_height = 0.02  # avoid table collision

        if config is not None:
            self.max_step = config.get("max_step", self.max_step)
            self.min_height = config.get("min_height", self.min_height)

    # ------------------------------------------------
    # Main interface
    # ------------------------------------------------
    def apply_action(self, action):
        """
        Apply abstract action to simulator.

        action = {
            "move": [dx, dy, dz],
            "grip": 0 | 1 | -1
        }
        """

        if action is None:
            return

        self._apply_move(action["move"])
        self._apply_grip(action["grip"])

    # ------------------------------------------------
    # Movement
    # ------------------------------------------------
    def _apply_move(self, delta):
        """
        Move end-effector by small delta (EE space).
        """
        delta = np.array(delta, dtype=float)

        # Clamp step size
        norm = np.linalg.norm(delta)
        if norm > self.max_step:
            delta = delta / norm * self.max_step

        # Get current EE pose
        ee_pos = self.simulator.get_ee_position()

        target_pos = ee_pos + delta

        # Safety: avoid going below table
        target_pos[2] = max(target_pos[2], self.min_height)

        self.simulator.set_ee_position(target_pos)

    # ------------------------------------------------
    # Gripper
    # ------------------------------------------------
    def _apply_grip(self, grip_cmd):
        """
        Control gripper opening / closing.

        grip_cmd:
            1  -> close
           -1  -> open
            0  -> no-op
        """
        if grip_cmd == 1:
            self.simulator.close_gripper()
        elif grip_cmd == -1:
            self.simulator.open_gripper()
        # else: do nothing
