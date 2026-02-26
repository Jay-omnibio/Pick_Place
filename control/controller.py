import numpy as np
import mujoco


class EEController:
    """
    End-effector controller with posture-aware IK.

    Primary task:
    - track EE Cartesian target from abstract action delta

    Secondary (nullspace) task:
    - bias joints toward a preferred posture to avoid stretched / awkward arm shapes
    """

    REQUIRED_CONFIG_KEYS = {
        "max_step",
        "min_height",
        "max_target_radius",
        "ik_damping",
        "nullspace_gain",
        "nullspace_gain_grasp",
        "max_joint_step",
        "ee_tolerance",
        "move_smoothing",
        "yaw_target",
        "yaw_weight",
        "yaw_weight_grasp",
        "enable_yaw_objective",
        "yaw_axis",
        "enable_topdown_objective",
        "topdown_weight",
        "topdown_weight_grasp",
        "tool_axis",
        "gripper_open_width",
        "gripper_close_width",
        "gripper_size_based_open",
        "gripper_open_clearance",
        "gripper_open_min_width",
        "gripper_open_unknown_full_open",
        "gripper_rate",
        "gripper_width_tol",
        "gripper_speed_tol",
        "gripper_switch_cooldown_steps",
        "debug_every_steps",
    }

    def __init__(self, simulator, config):
        self.simulator = simulator
        if not isinstance(config, dict):
            raise ValueError("EEController requires a config dict loaded from config/common_robot.yaml.")
        missing = sorted(self.REQUIRED_CONFIG_KEYS - set(config.keys()))
        if missing:
            raise ValueError(f"EEController config missing required keys: {missing}")
        extra = sorted(set(config.keys()) - self.REQUIRED_CONFIG_KEYS)
        if extra:
            raise ValueError(f"EEController config has unknown keys: {extra}")

        # Action-space and safety limits
        self.max_step = float(config["max_step"])
        self.min_height = float(config["min_height"])
        self.max_target_radius = float(config["max_target_radius"])

        # IK / posture parameters
        self.ik_damping = float(config["ik_damping"])
        self.nullspace_gain = float(config["nullspace_gain"])
        self.nullspace_gain_grasp = float(config["nullspace_gain_grasp"])
        self.max_joint_step = float(config["max_joint_step"])
        self.ee_tolerance = float(config["ee_tolerance"])
        self.move_smoothing = float(config["move_smoothing"])

        # Orientation objective (yaw alignment in world XY plane)
        # IMPORTANT:
        # - If yaw_target is an int in {0,1,2,3}, it's treated as a cardinal direction:
        #     0:+X, 1:+Y, 2:-X, 3:-Y
        # - Otherwise it is treated as a yaw angle in radians.
        # This avoids the previous bug where "3" was interpreted as 3 radians (~171°).
        self.yaw_target = config["yaw_target"]
        self.yaw_weight = float(config["yaw_weight"])
        self.yaw_weight_grasp = float(config["yaw_weight_grasp"])
        self.enable_yaw_objective = bool(config["enable_yaw_objective"])
        self.yaw_axis = int(config["yaw_axis"])
        self.enable_topdown_objective = bool(config["enable_topdown_objective"])
        self.topdown_weight = float(config["topdown_weight"])
        self.topdown_weight_grasp = float(config["topdown_weight_grasp"])
        self.tool_axis = int(config["tool_axis"])

        # Gripper dynamics
        self.gripper_open_width = float(config["gripper_open_width"])
        self.gripper_close_width = float(config["gripper_close_width"])
        self.gripper_size_based_open = bool(config["gripper_size_based_open"])
        self.gripper_open_clearance = max(0.0, float(config["gripper_open_clearance"]))
        self.gripper_open_min_width = float(config["gripper_open_min_width"])
        self.gripper_open_unknown_full_open = bool(config["gripper_open_unknown_full_open"])
        self.gripper_rate = float(config["gripper_rate"])
        self.gripper_target_width = None 
        self.gripper_mode = "open"
        self.gripper_state = "READY"
        self.gripper_width_tol = float(config["gripper_width_tol"])
        self.gripper_speed_tol = float(config["gripper_speed_tol"])
        self.gripper_switch_cooldown_steps = int(config["gripper_switch_cooldown_steps"])
        self.gripper_switch_cooldown = 0 
        self.debug_every_steps = int(config["debug_every_steps"])
        self.control_step = 0
        self.last_requested_move_norm = 0.0
        self.last_applied_move_norm = 0.0
        self.last_dq_norm_raw = 0.0
        self.last_dq_norm_applied = 0.0
        self.prev_move_delta = np.zeros(3, dtype=float)
        self.gripper_object_width_estimate = None

        self.gripper_open_min_width = float(
            np.clip(self.gripper_open_min_width, self.gripper_close_width, self.gripper_open_width)
        )
        self.gripper_object_width_estimate = self._estimate_object_grasp_width()

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
            optional: "yaw_target": float (radians) or cardinal int {0,1,2,3},
            optional: "yaw_pi_symmetric": bool (treat yaw and yaw+pi as equivalent; pick shorter turn),
            optional: "enable_topdown_objective": bool,
            optional: "position_gain_scale": float (>0),
            optional: "yaw_weight_scale": float (>0),
            optional: "topdown_weight_scale": float (>0),
            optional: "nullspace_gain_scale": float (>0),
        }
        If ee_target_pos is given, move is computed as (ee_target_pos - current_ee), clamped by max_step * max_step_scale.
        """
        if action is None:
            return

        grip_cmd = int(action.get("grip", 0))
        close_target_override = action.get("grip_close_target_width", None)
        open_target_override = action.get("grip_open_target_width", None)
        # Per-step orientation (do not overwrite instance defaults).
        self._step_enable_yaw = action.get("enable_yaw_objective", self.enable_yaw_objective)
        self._step_yaw_target = self._interpret_yaw_target(action.get("yaw_target", self.yaw_target))
        self._step_yaw_pi_symmetric = bool(action.get("yaw_pi_symmetric", False))
        self._step_enable_topdown = action.get("enable_topdown_objective", self.enable_topdown_objective)
        # Optional per-step gain scheduling (useful for softer grasp/place phases).
        self._step_position_gain_scale = float(np.clip(action.get("position_gain_scale", 1.0), 0.05, 2.0))
        self._step_yaw_weight_scale = float(np.clip(action.get("yaw_weight_scale", 1.0), 0.05, 2.0))
        self._step_topdown_weight_scale = float(np.clip(action.get("topdown_weight_scale", 1.0), 0.05, 2.0))
        self._step_nullspace_gain_scale = float(np.clip(action.get("nullspace_gain_scale", 1.0), 0.05, 2.0))

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
        self._apply_grip(
            grip_cmd,
            close_target_override=close_target_override,
            open_target_override=open_target_override,
        )
        self.control_step += 1
        if self.debug_every_steps > 0 and self.control_step % self.debug_every_steps == 0:
            print(
                f"[CtrlDbg] step={self.control_step} "
                f"req_move={self.last_requested_move_norm:.4f} "
                f"applied_move={self.last_applied_move_norm:.4f} max_step={self.max_step:.4f} "
                f"dq_raw={self.last_dq_norm_raw:.4f} dq_applied={self.last_dq_norm_applied:.4f} "
                f"max_joint_step={self.max_joint_step:.4f} "
                f"scales=pos:{self._step_position_gain_scale:.2f},yaw:{self._step_yaw_weight_scale:.2f},"
                f"top:{self._step_topdown_weight_scale:.2f},null:{self._step_nullspace_gain_scale:.2f} "
                f"gripper={self.gripper_state}/{self.gripper_mode}"
            )

    def _apply_move(self, delta, grip_cmd=0):
        delta = np.array(delta, dtype=float).reshape(3,)
        if not np.all(np.isfinite(delta)):
            return
        gain = float(getattr(self, "_step_position_gain_scale", 1.0))
        delta = gain * delta
        # Real robots use velocity smoothing/limits to avoid jerky path changes.
        alpha = float(np.clip(self.move_smoothing, 0.0, 0.95))
        prev = self.prev_move_delta.copy()
        # If an axis command goes to zero or flips sign, clear smoothing memory
        # on that axis so stale momentum does not push the wrong way.
        axis_eps = 1e-6
        for i in range(3):
            cmd_i = float(delta[i])
            prev_i = float(prev[i])
            if abs(cmd_i) <= axis_eps:
                prev[i] = 0.0
            elif abs(prev_i) > axis_eps and np.sign(cmd_i) != np.sign(prev_i):
                prev[i] = 0.0
        delta = alpha * prev + (1.0 - alpha) * delta
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
            top_w = float(top_w) * float(getattr(self, "_step_topdown_weight_scale", 1.0))
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
            yaw_target = float(getattr(self, "_step_yaw_target", self.yaw_target))
            yaw_err = self._compute_yaw_error(
                current_yaw=ee_yaw,
                desired_yaw=yaw_target,
                pi_symmetric=bool(getattr(self, "_step_yaw_pi_symmetric", False)),
            )
            yaw_w = self.yaw_weight_grasp if grip_cmd == 1 else self.yaw_weight
            yaw_w = float(yaw_w) * float(getattr(self, "_step_yaw_weight_scale", 1.0))
            if yaw_w > 0.0:
                J_yaw = jacr[2, self.arm_dof_addr].reshape(1, -1)
                J = np.vstack([J, yaw_w * J_yaw])
                err = np.concatenate([err, np.array([yaw_w * yaw_err])])

        q = data.qpos[self.arm_qpos_addr].copy()
        null_gain = self.nullspace_gain_grasp if grip_cmd == 1 else self.nullspace_gain
        null_gain = float(null_gain) * float(getattr(self, "_step_nullspace_gain_scale", 1.0))
        q_err = self.q_pref - q

        def _solve_dq(J_use, err_use):
            JT = J_use.T
            task_dim = J_use.shape[0]
            A = J_use @ JT + (self.ik_damping ** 2) * np.eye(task_dim)
            try:
                # Damped least-squares pseudoinverse via linear solve
                A_inv_err = np.linalg.solve(A, err_use)
                dq_task = JT @ A_inv_err

                A_inv_I = np.linalg.solve(A, np.eye(task_dim))
                J_pinv = JT @ A_inv_I
            except np.linalg.LinAlgError:
                return None
            dq_null = null_gain * q_err
            N = np.eye(len(q)) - (J_pinv @ J_use)
            return dq_task + N @ dq_null

        dq = _solve_dq(J, err)
        if dq is None:
            # Rare numerical issue: fallback to mocap path
            self.simulator.set_ee_position(target_pos)
            return

        # Safety guard: if full objective predicts XY motion away from target,
        # solve this step with translation-only objective.
        xy_demand = float(np.linalg.norm(err_pos[:2]))
        if xy_demand > 0.004:
            pred_delta = J_pos @ dq
            if float(np.dot(pred_delta[:2], err_pos[:2])) < -1e-6:
                dq_pos = _solve_dq(J_pos, err_pos)
                if dq_pos is not None:
                    dq = dq_pos

        dq_norm = np.linalg.norm(dq)
        self.last_dq_norm_raw = float(dq_norm)
        if dq_norm > self.max_joint_step and dq_norm > 0:
            dq = (dq / dq_norm) * self.max_joint_step
        self.last_dq_norm_applied = float(np.linalg.norm(dq))

        q_target = q + dq
        q_target = np.clip(q_target, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

        data.ctrl[self.arm_actuator_ids] = q_target
        mujoco.mj_forward(model, data)

    def _apply_grip(self, grip_cmd, close_target_override=None, open_target_override=None):
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

        # Important: allow close -> open switch immediately, even when close is not
        # "READY" (object can block full close, which otherwise deadlocks release).
        can_switch = (self.gripper_switch_cooldown == 0) and (
            (self.gripper_state == "READY")
            or (requested_mode == "open" and self.gripper_mode == "close")
        )
        if requested_mode is not None and requested_mode != self.gripper_mode and can_switch:
            self.gripper_mode = requested_mode
            self.gripper_switch_cooldown = self.gripper_switch_cooldown_steps

        close_target = self.gripper_close_width
        if close_target_override is not None:
            try:
                close_target = float(close_target_override)
            except (TypeError, ValueError):
                close_target = self.gripper_close_width
            if not np.isfinite(close_target):
                close_target = self.gripper_close_width
            close_target = float(np.clip(close_target, self.gripper_close_width, self.gripper_open_width))

        open_target = self._default_open_target_width()
        if open_target_override is not None:
            try:
                open_target = float(open_target_override)
            except (TypeError, ValueError):
                open_target = self._default_open_target_width()
            if not np.isfinite(open_target):
                open_target = self._default_open_target_width()
            open_target = float(np.clip(open_target, self.gripper_close_width, self.gripper_open_width))

        self.gripper_target_width = close_target if self.gripper_mode == "close" else open_target

        diff = self.gripper_target_width - current_width
        if abs(diff) < self.gripper_rate:
            next_width = self.gripper_target_width
        else:
            next_width = current_width + np.sign(diff) * self.gripper_rate
        self.simulator.set_gripper_width(next_width)

    def _estimate_object_grasp_width(self):
        getter = getattr(self.simulator, "get_object_grasp_width_estimate", None)
        if not callable(getter):
            return None
        try:
            width = float(getter())
        except Exception:
            return None
        if not np.isfinite(width) or width <= 0.0:
            return None
        return float(width)

    def get_default_open_target_width(self):
        return float(self._default_open_target_width())

    def _default_open_target_width(self):
        # Legacy behavior: always open to configured max width.
        if not self.gripper_size_based_open:
            return float(self.gripper_open_width)

        # Refresh object width estimate continuously so open target adapts if
        # object changes (shape/pose) between episodes or retries.
        latest_width = self._estimate_object_grasp_width()
        if latest_width is not None:
            self.gripper_object_width_estimate = latest_width

        obj_width = self.gripper_object_width_estimate

        if obj_width is None:
            if self.gripper_open_unknown_full_open:
                return float(self.gripper_open_width)
            return float(self.gripper_open_min_width)

        target = float(obj_width + self.gripper_open_clearance)
        return float(np.clip(target, self.gripper_open_min_width, self.gripper_open_width))

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

    @classmethod
    def _compute_yaw_error(cls, current_yaw, desired_yaw, pi_symmetric=False):
        """
        Compute shortest signed yaw error.
        If pi_symmetric=True, treat yaw and yaw+pi as equivalent and choose
        the smallest-magnitude error among equivalent targets.
        """
        e0 = cls._wrap_to_pi(float(desired_yaw) - float(current_yaw))
        if not pi_symmetric:
            return e0
        e1 = cls._wrap_to_pi(float(desired_yaw + np.pi) - float(current_yaw))
        e2 = cls._wrap_to_pi(float(desired_yaw - np.pi) - float(current_yaw))
        return min((e0, e1, e2), key=lambda x: abs(float(x)))

    @staticmethod
    def _wrap_to_pi(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
