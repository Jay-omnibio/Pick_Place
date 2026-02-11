import numpy as np


class EEController:
    """
    End-effector level controller.

    Converts abstract actions into low-level commands.
    """

    def __init__(self, simulator, config=None):
        self.simulator = simulator

        # Safety limits
        self.max_step = 0.012   # meters per step
        self.min_height = 0.02  # avoid table collision
        self.pregrasp_height = 0.10
        self.xy_descend_gate = 0.07
        self.xy_lock_gate = 0.03
        self.lateral_couple_gain = 0.25
        self.max_lateral_couple_step = 0.004
        self.max_radial_reach = 0.62
        self.radial_recovery_step = 0.003

        if config is not None:
            self.max_step = config.get("max_step", self.max_step)
            self.min_height = config.get("min_height", self.min_height)
            self.pregrasp_height = config.get("pregrasp_height", self.pregrasp_height)
            self.xy_descend_gate = config.get("xy_descend_gate", self.xy_descend_gate)
            self.xy_lock_gate = config.get("xy_lock_gate", self.xy_lock_gate)
            self.lateral_couple_gain = config.get("lateral_couple_gain", self.lateral_couple_gain)
            self.max_lateral_couple_step = config.get("max_lateral_couple_step", self.max_lateral_couple_step)
            self.max_radial_reach = config.get("max_radial_reach", self.max_radial_reach)
            self.radial_recovery_step = config.get("radial_recovery_step", self.radial_recovery_step)

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

        # Integrate commands in mocap space; EE can lag behind weld tracking.
        if hasattr(self.simulator, "get_mocap_position"):
            base_pos = self.simulator.get_mocap_position()
        else:
            base_pos = self.simulator.get_ee_position()

        target_pos = base_pos + delta
        target_pos = self._shape_target_for_posture(target_pos, delta)

        # Safety: avoid going below table
        target_pos[2] = max(target_pos[2], self.min_height)

        self.simulator.set_ee_position(target_pos)

    def _shape_target_for_posture(self, target_pos, delta):
        """
        Posture-aware shaping layer:
        - keep an above-object pregrasp height while XY is not aligned
        - couple downward intent with lateral correction toward object
        - add mild radial anti-stretch bias near kinematic edge
        """
        target_pos = np.array(target_pos, dtype=float)
        ee_pos = self.simulator.get_ee_position()
        obj_pos = self.simulator.get_object_position()

        xy_err = obj_pos[:2] - ee_pos[:2]
        xy_dist = np.linalg.norm(xy_err)
        descending = delta[2] < 0.0

        # Do not descend aggressively before XY is close to object.
        pregrasp_z = obj_pos[2] + self.pregrasp_height
        if xy_dist > self.xy_descend_gate:
            target_pos[2] = max(target_pos[2], pregrasp_z)

        # If trying to descend while still misaligned, blend in XY correction.
        if descending and xy_dist > self.xy_lock_gate:
            lateral = self.lateral_couple_gain * xy_err
            lateral_norm = np.linalg.norm(lateral)
            if lateral_norm > self.max_lateral_couple_step and lateral_norm > 0:
                lateral = lateral / lateral_norm * self.max_lateral_couple_step
            target_pos[:2] += lateral

        # Avoid stretched-arm descent near edge of reachability.
        radial = np.linalg.norm(ee_pos[:2])
        if descending and radial > self.max_radial_reach and radial > 1e-8:
            inward = -ee_pos[:2] / radial
            target_pos[:2] += inward * self.radial_recovery_step
            target_pos[2] = max(target_pos[2], ee_pos[2])

        return target_pos

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
