import numpy as np
import mujoco
import os


class EEController:
    """
    End-effector controller with posture-aware IK.

    Primary task:
    - track EE Cartesian target from abstract action delta

    Secondary (nullspace) task:
    - bias joints toward a preferred posture to avoid stretched / awkward arm shapes
    """

    def __init__(self, simulator, config=None):
        self.simulator = simulator

        # Action-space and safety limits
        self.max_step = 0.03  #0.015 max EE movement per control step, to avoid large jumps that may cause instability
        self.min_height = 0.02
        self.max_target_radius = 0.78

        # IK / posture parameters
        self.ik_damping = 0.03
        self.nullspace_gain = 0.15
        self.nullspace_gain_grasp = 0.05
        self.max_joint_step = 0.20 #0.10 per control step, to avoid large jumps that may cause instability
        self.ee_tolerance = 1e-4
        self.move_smoothing = 0.55

        # Orientation objective (yaw alignment in world XY plane)
        # IMPORTANT:
        # - If yaw_target is an int in {0,1,2,3}, it's treated as a cardinal direction:
        #     0:+X, 1:+Y, 2:-X, 3:-Y
        # - Otherwise it is treated as a yaw angle in radians.
        # This avoids the previous bug where "3" was interpreted as 3 radians (~171°).
        self.yaw_target = 3
        self.yaw_weight = 0.25 # orientation objective weight during approach
        self.yaw_weight_grasp = 1.0 # orientation objective weight during grasp (when grip_cmd=1), set higher to prioritize correct gripper orientation for grasping
        self.enable_yaw_objective = (os.getenv("CTRL_ENABLE_YAW", "1") == "1")
        self.yaw_axis = 1 # gripper Y axis should align with world Y for best grasping, but can be set to 0 (X) or 2 (Z) for different approach orientations
        self.enable_topdown_objective = (os.getenv("CTRL_ENABLE_TOPDOWN", "1") == "1")
        self.topdown_weight = 0.60   # stronger during approach so gripper stays vertical
        self.topdown_weight_grasp = 1.20  # even stronger during descend/close to avoid flat orientation
        self.tool_axis = 2 #

        # Gripper dynamics
        self.gripper_open_width = 0.060
        self.gripper_close_width = 0.0
        self.gripper_rate = 0.03 # how much the gripper width can change per control step, to avoid instability changed form 0.03
        self.gripper_target_width = None
        self.gripper_mode = "open"
        self.gripper_state = "READY"
        self.gripper_width_tol = 0.0015
        self.gripper_speed_tol = 0.002
        self.gripper_switch_cooldown_steps = 6
        self.gripper_switch_cooldown = 0
        self.debug_every_steps = int(os.getenv("CTRL_DEBUG_EVERY_STEPS", "100"))
        self.control_step = 0
        self.last_requested_move_norm = 0.0
        self.last_applied_move_norm = 0.0
        self.last_dq_norm_raw = 0.0
        self.last_dq_norm_applied = 0.0
        self.prev_move_delta = np.zeros(3, dtype=float)

        if config is not None:
            self.max_step = config.get("max_step", self.max_step)
            self.min_height = config.get("min_height", self.min_height)
            self.max_target_radius = config.get("max_target_radius", self.max_target_radius)
            self.ik_damping = config.get("ik_damping", self.ik_damping)
            self.nullspace_gain = config.get("nullspace_gain", self.nullspace_gain)
            self.nullspace_gain_grasp = config.get("nullspace_gain_grasp", self.nullspace_gain_grasp)
            self.max_joint_step = config.get("max_joint_step", self.max_joint_step)
            self.ee_tolerance = config.get("ee_tolerance", self.ee_tolerance)
            self.move_smoothing = config.get("move_smoothing", self.move_smoothing)
            self.yaw_target = config.get("yaw_target", self.yaw_target)
            self.yaw_weight = config.get("yaw_weight", self.yaw_weight)
            self.yaw_weight_grasp = config.get("yaw_weight_grasp", self.yaw_weight_grasp)
            self.enable_yaw_objective = config.get("enable_yaw_objective", self.enable_yaw_objective)
            self.yaw_axis = int(config.get("yaw_axis", self.yaw_axis))
            self.enable_topdown_objective = config.get("enable_topdown_objective", self.enable_topdown_objective)
            self.topdown_weight = config.get("topdown_weight", self.topdown_weight)
            self.topdown_weight_grasp = config.get("topdown_weight_grasp", self.topdown_weight_grasp)
            self.tool_axis = int(config.get("tool_axis", self.tool_axis))
            self.gripper_open_width = config.get("gripper_open_width", self.gripper_open_width)
            self.gripper_close_width = config.get("gripper_close_width", self.gripper_close_width)
            self.gripper_rate = config.get("gripper_rate", self.gripper_rate)
            self.gripper_width_tol = config.get("gripper_width_tol", self.gripper_width_tol)
            self.gripper_speed_tol = config.get("gripper_speed_tol", self.gripper_speed_tol)
            self.gripper_switch_cooldown_steps = config.get(
                "gripper_switch_cooldown_steps", self.gripper_switch_cooldown_steps
            )
            self.debug_every_steps = int(config.get("debug_every_steps", self.debug_every_steps))

        # Environment variables override config (useful for quick experiments).
        if os.getenv("CTRL_ENABLE_YAW") is not None:
            self.enable_yaw_objective = (os.getenv("CTRL_ENABLE_YAW", "1") == "1")
        if os.getenv("CTRL_ENABLE_TOPDOWN") is not None:
            self.enable_topdown_objective = (os.getenv("CTRL_ENABLE_TOPDOWN", "1") == "1")

        self.yaw_target = self._interpret_yaw_target(self.yaw_target)

        self.use_ik = False
        self.arm_joint_ids = None
        self.arm_qpos_addr = None
        self.arm_dof_addr = None
        self.arm_ranges = None
        self.arm_actuator_ids = None
        self.q_pref = None
        # Elbow-up nominal Panda posture to avoid side-collapsed shapes near grasp.
        self.preferred_posture = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.8], dtype=float)
        self._init_ik()

    def _init_ik(self):
        """
        Discover arm joints/actuators required for IK.
        If anything is missing, keep fallback mocap control.
        """
        model = self.simulator.model
        data = self.simulator.data

        joint_names = [f"joint{i}" for i in range(1, 8)]
        actuator_names = [f"actuator{i}" for i in range(1, 8)]

        joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                return
            joint_ids.append(jid)

        actuator_ids = []
        for name in actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                return
            actuator_ids.append(aid)

        self.arm_joint_ids = np.array(joint_ids, dtype=int)
        self.arm_qpos_addr = model.jnt_qposadr[self.arm_joint_ids].astype(int)
        self.arm_dof_addr = model.jnt_dofadr[self.arm_joint_ids].astype(int)
        self.arm_ranges = model.jnt_range[self.arm_joint_ids].copy()
        self.arm_actuator_ids = np.array(actuator_ids, dtype=int)
        if self.preferred_posture.shape[0] == self.arm_qpos_addr.shape[0]:
            self.q_pref = np.clip(self.preferred_posture, self.arm_ranges[:, 0], self.arm_ranges[:, 1])
        else:
            self.q_pref = data.qpos[self.arm_qpos_addr].copy()
        self._disable_mocap_weld_if_present()
        self.use_ik = True

    def _disable_mocap_weld_if_present(self):
        """
        The provided model welds EE to mocap for mocap-driven control.
        Disable that weld when IK mode is active, otherwise joint IK cannot move EE.
        """
        model = self.simulator.model
        data = self.simulator.data

        mocap_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_mocap")
        ee_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ee_center_body")
        if mocap_body < 0 or ee_body < 0 or model.neq <= 0:
            return

        for i in range(model.neq):
            if model.eq_type[i] != mujoco.mjtEq.mjEQ_WELD:
                continue
            b1 = int(model.eq_obj1id[i])
            b2 = int(model.eq_obj2id[i])
            if (b1 == mocap_body and b2 == ee_body) or (b1 == ee_body and b2 == mocap_body):
                data.eq_active[i] = 0
        mujoco.mj_forward(model, data)

    def apply_action(self, action):
        """
        action = {
            "move": [dx, dy, dz]  OR  "ee_target_pos": [x, y, z] (world frame),
            "grip": 0|1|-1,
            optional: "max_step_scale": float (0..1 for guarded motion),
            optional: "enable_yaw_objective": bool,
            optional: "enable_topdown_objective": bool,
        }
        If ee_target_pos is given, move is computed as (ee_target_pos - current_ee), clamped by max_step * max_step_scale.
        """
        if action is None:
            return

        grip_cmd = int(action.get("grip", 0))
        # Per-step orientation (do not overwrite instance defaults).
        self._step_enable_yaw = action.get("enable_yaw_objective", self.enable_yaw_objective)
        self._step_enable_topdown = action.get("enable_topdown_objective", self.enable_topdown_objective)

        if "ee_target_pos" in action:
            ee = self.simulator.get_ee_position()
            delta = np.array(action["ee_target_pos"], dtype=float).reshape(3,) - ee
            if np.all(np.isfinite(delta)):
                scale = float(np.clip(action.get("max_step_scale", 1.0), 0.01, 1.0))
                max_effective = self.max_step * scale
                n = np.linalg.norm(delta)
                if n > max_effective and n > 0:
                    delta = (delta / n) * max_effective
                move = delta.tolist()
            else:
                move = [0.0, 0.0, 0.0]
        else:
            move = action.get("move", [0.0, 0.0, 0.0])
        self._apply_move(move, grip_cmd=grip_cmd)
        self._apply_grip(grip_cmd)
        self.control_step += 1
        if self.debug_every_steps > 0 and self.control_step % self.debug_every_steps == 0:
            print(
                f"[CtrlDbg] step={self.control_step} "
                f"req_move={self.last_requested_move_norm:.4f} "
                f"applied_move={self.last_applied_move_norm:.4f} max_step={self.max_step:.4f} "
                f"dq_raw={self.last_dq_norm_raw:.4f} dq_applied={self.last_dq_norm_applied:.4f} "
                f"max_joint_step={self.max_joint_step:.4f} "
                f"gripper={self.gripper_state}/{self.gripper_mode}"
            )

    def _apply_move(self, delta, grip_cmd=0):
        delta = np.array(delta, dtype=float).reshape(3,)
        if not np.all(np.isfinite(delta)):
            return
        # Real robots use velocity smoothing/limits to avoid jerky path changes.
        alpha = float(np.clip(self.move_smoothing, 0.0, 0.95))
        delta = alpha * self.prev_move_delta + (1.0 - alpha) * delta
        self.prev_move_delta = delta.copy()
        self.last_requested_move_norm = float(np.linalg.norm(delta))
        n = np.linalg.norm(delta)
        if n > self.max_step and n > 0:
            delta = (delta / n) * self.max_step

        ee_pos = self.simulator.get_ee_position()
        target_pos = ee_pos + delta
        target_pos[2] = max(target_pos[2], self.min_height)

        radial = np.linalg.norm(target_pos[:2])
        if radial > self.max_target_radius and radial > 0:
            target_pos[:2] = target_pos[:2] / radial * self.max_target_radius
        self.last_applied_move_norm = float(np.linalg.norm(target_pos - ee_pos))

        if self.use_ik:
            self._apply_ik_target(target_pos, grip_cmd=grip_cmd)
        else:
            # Fallback for models without arm actuator/joint naming.
            self.simulator.set_ee_position(target_pos)

    def _apply_ik_target(self, target_pos, grip_cmd=0):
        model = self.simulator.model
        data = self.simulator.data

        ee_pos = self.simulator.get_ee_position()
        err_pos = np.array(target_pos, dtype=float) - ee_pos
        if np.linalg.norm(err_pos) < self.ee_tolerance:
            return

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, self.simulator.ee_site_id)

        J_pos = jacp[:, self.arm_dof_addr]
        J = J_pos
        err = err_pos

        # Prefer approaching from above: align tool axis with world down.
        if getattr(self, "_step_enable_topdown", self.enable_topdown_objective):
            top_w = self.topdown_weight_grasp if grip_cmd == 1 else self.topdown_weight
            if top_w > 0.0:
                ee_xmat = self._get_ee_xmat()
                axis_idx = int(np.clip(self.tool_axis, 0, 2))
                tool_dir = ee_xmat[:, axis_idx]
                desired_down = np.array([0.0, 0.0, -1.0], dtype=float)
                tilt_err = np.cross(tool_dir, desired_down)
                J_top = jacr[0:2, self.arm_dof_addr]
                J = np.vstack([J, top_w * J_top])
                err = np.concatenate([err, top_w * tilt_err[0:2]])

        if getattr(self, "_step_enable_yaw", self.enable_yaw_objective):
            ee_yaw = self._get_ee_yaw()
            yaw_err = self._wrap_to_pi(self.yaw_target - ee_yaw)
            yaw_w = self.yaw_weight_grasp if grip_cmd == 1 else self.yaw_weight
            if yaw_w > 0.0:
                J_yaw = jacr[2, self.arm_dof_addr].reshape(1, -1)
                J = np.vstack([J, yaw_w * J_yaw])
                err = np.concatenate([err, np.array([yaw_w * yaw_err])])

        JT = J.T
        task_dim = J.shape[0]
        A = J @ JT + (self.ik_damping ** 2) * np.eye(task_dim)

        try:
            # Damped least-squares pseudoinverse via linear solve
            A_inv_err = np.linalg.solve(A, err)
            dq_task = JT @ A_inv_err

            A_inv_I = np.linalg.solve(A, np.eye(task_dim))
            J_pinv = JT @ A_inv_I
        except np.linalg.LinAlgError:
            # Rare numerical issue: fallback to mocap path
            self.simulator.set_ee_position(target_pos)
            return

        q = data.qpos[self.arm_qpos_addr].copy()
        null_gain = self.nullspace_gain_grasp if grip_cmd == 1 else self.nullspace_gain
        dq_null = null_gain * (self.q_pref - q)
        N = np.eye(len(q)) - (J_pinv @ J)
        dq = dq_task + N @ dq_null

        dq_norm = np.linalg.norm(dq)
        self.last_dq_norm_raw = float(dq_norm)
        if dq_norm > self.max_joint_step and dq_norm > 0:
            dq = (dq / dq_norm) * self.max_joint_step
        self.last_dq_norm_applied = float(np.linalg.norm(dq))

        q_target = q + dq
        q_target = np.clip(q_target, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

        data.ctrl[self.arm_actuator_ids] = q_target
        mujoco.mj_forward(model, data)

    def _apply_grip(self, grip_cmd):
        current_width = float(self.simulator.get_gripper_width())
        current_speed = float(abs(self.simulator.get_gripper_speed()))

        if self.gripper_target_width is None:
            self.gripper_target_width = current_width

        # Update READY/MOVING state from measured width + velocity.
        width_err = abs(self.gripper_target_width - current_width)
        is_ready = (width_err <= self.gripper_width_tol) and (current_speed <= self.gripper_speed_tol)
        self.gripper_state = "READY" if is_ready else "MOVING"

        # Latch request, but only switch open<->close when ready or cooldown elapsed.
        requested_mode = None
        if grip_cmd == 1:
            requested_mode = "close"
        elif grip_cmd == -1:
            requested_mode = "open"

        if self.gripper_switch_cooldown > 0:
            self.gripper_switch_cooldown -= 1

        can_switch = (self.gripper_state == "READY") and (self.gripper_switch_cooldown == 0)
        if requested_mode is not None and requested_mode != self.gripper_mode and can_switch:
            self.gripper_mode = requested_mode
            self.gripper_switch_cooldown = self.gripper_switch_cooldown_steps

        self.gripper_target_width = (
            self.gripper_close_width if self.gripper_mode == "close" else self.gripper_open_width
        )

        diff = self.gripper_target_width - current_width
        if abs(diff) < self.gripper_rate:
            next_width = self.gripper_target_width
        else:
            next_width = current_width + np.sign(diff) * self.gripper_rate
        self.simulator.set_gripper_width(next_width)

    def _get_ee_yaw(self):
        xmat = self._get_ee_xmat()
        axis_index = int(np.clip(self.yaw_axis, 0, 2))
        v = xmat[:, axis_index]
        return float(np.arctan2(v[1], v[0]))

    @staticmethod
    def _interpret_yaw_target(yaw_target):
        """
        Convert yaw_target to radians if given as a cardinal int in {0,1,2,3}.
        """
        try:
            # numpy ints also handled by int(...)
            if isinstance(yaw_target, (int, np.integer)) and int(yaw_target) in (0, 1, 2, 3):
                idx = int(yaw_target)
                mapping = {
                    0: 0.0,            # +X
                    1: np.pi / 2.0,    # +Y
                    2: np.pi,          # -X
                    3: -np.pi / 2.0,   # -Y
                }
                return float(mapping[idx])
        except Exception:
            pass
        return float(yaw_target)

    def _get_ee_xmat(self):
        return np.array(self.simulator.data.site_xmat[self.simulator.ee_site_id], dtype=float).reshape(3, 3)

    @staticmethod
    def _wrap_to_pi(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
