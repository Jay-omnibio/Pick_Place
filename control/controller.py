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

    def __init__(self, simulator, config=None):
        self.simulator = simulator

        # Action-space and safety limits
        self.max_step = 0.03  #0.015 max EE movement per control step, to avoid large jumps that may cause instability
        self.min_height = 0.02
        self.max_target_radius = 0.78

        # IK / posture parameters
        self.ik_damping = 0.03
        self.nullspace_gain = 0.15
        self.max_joint_step = 0.20 #0.10 per control step, to avoid large jumps that may cause instability
        self.ee_tolerance = 1e-4

        # Orientation objective (yaw alignment in world XY plane)
        self.yaw_target = 0.0
        self.yaw_weight = 0.25
        self.yaw_weight_grasp = 1.0
        self.enable_yaw_objective = True
        self.yaw_axis = 1

        # Gripper dynamics
        self.gripper_open_width = 0.040
        self.gripper_close_width = 0.0
        self.gripper_rate = 0.003
        self.gripper_target_width = None

        if config is not None:
            self.max_step = config.get("max_step", self.max_step)
            self.min_height = config.get("min_height", self.min_height)
            self.max_target_radius = config.get("max_target_radius", self.max_target_radius)
            self.ik_damping = config.get("ik_damping", self.ik_damping)
            self.nullspace_gain = config.get("nullspace_gain", self.nullspace_gain)
            self.max_joint_step = config.get("max_joint_step", self.max_joint_step)
            self.ee_tolerance = config.get("ee_tolerance", self.ee_tolerance)
            self.yaw_target = config.get("yaw_target", self.yaw_target)
            self.yaw_weight = config.get("yaw_weight", self.yaw_weight)
            self.yaw_weight_grasp = config.get("yaw_weight_grasp", self.yaw_weight_grasp)
            self.enable_yaw_objective = config.get("enable_yaw_objective", self.enable_yaw_objective)
            self.yaw_axis = int(config.get("yaw_axis", self.yaw_axis))
            self.gripper_open_width = config.get("gripper_open_width", self.gripper_open_width)
            self.gripper_close_width = config.get("gripper_close_width", self.gripper_close_width)
            self.gripper_rate = config.get("gripper_rate", self.gripper_rate)

        self.use_ik = False
        self.arm_joint_ids = None
        self.arm_qpos_addr = None
        self.arm_dof_addr = None
        self.arm_ranges = None
        self.arm_actuator_ids = None
        self.q_pref = None
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
        action = {"move": [dx, dy, dz], "grip": 0|1|-1}
        """
        if action is None:
            return

        grip_cmd = int(action.get("grip", 0))
        self._apply_move(action["move"], grip_cmd=grip_cmd)
        self._apply_grip(grip_cmd)

    def _apply_move(self, delta, grip_cmd=0):
        delta = np.array(delta, dtype=float)
        n = np.linalg.norm(delta)
        if n > self.max_step and n > 0:
            delta = (delta / n) * self.max_step

        ee_pos = self.simulator.get_ee_position()
        target_pos = ee_pos + delta
        target_pos[2] = max(target_pos[2], self.min_height)

        radial = np.linalg.norm(target_pos[:2])
        if radial > self.max_target_radius and radial > 0:
            target_pos[:2] = target_pos[:2] / radial * self.max_target_radius

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

        if self.enable_yaw_objective:
            ee_yaw = self._get_ee_yaw()
            yaw_err = self._wrap_to_pi(self.yaw_target - ee_yaw)
            yaw_w = self.yaw_weight_grasp if grip_cmd == 1 else self.yaw_weight
            if yaw_w > 0.0:
                J_yaw = jacr[2, self.arm_dof_addr].reshape(1, -1)
                J = np.vstack([J_pos, yaw_w * J_yaw])
                err = np.concatenate([err_pos, np.array([yaw_w * yaw_err])])

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
        dq_null = self.nullspace_gain * (self.q_pref - q)
        N = np.eye(len(q)) - (J_pinv @ J)
        dq = dq_task + N @ dq_null

        dq_norm = np.linalg.norm(dq)
        if dq_norm > self.max_joint_step and dq_norm > 0:
            dq = (dq / dq_norm) * self.max_joint_step

        q_target = q + dq
        q_target = np.clip(q_target, self.arm_ranges[:, 0], self.arm_ranges[:, 1])

        data.ctrl[self.arm_actuator_ids] = q_target
        mujoco.mj_forward(model, data)

    def _apply_grip(self, grip_cmd):
        if self.gripper_target_width is None:
            self.gripper_target_width = float(self.simulator.get_gripper_width())

        if grip_cmd == 1:
            self.gripper_target_width = self.gripper_close_width
        elif grip_cmd == -1:
            self.gripper_target_width = self.gripper_open_width

        current_width = float(self.simulator.get_gripper_width())
        diff = self.gripper_target_width - current_width
        if abs(diff) < self.gripper_rate:
            next_width = self.gripper_target_width
        else:
            next_width = current_width + np.sign(diff) * self.gripper_rate
        self.simulator.set_gripper_width(next_width)

    def _get_ee_yaw(self):
        xmat = np.array(self.simulator.data.site_xmat[self.simulator.ee_site_id], dtype=float).reshape(3, 3)
        axis_index = int(np.clip(self.yaw_axis, 0, 2))
        v = xmat[:, axis_index]
        return float(np.arctan2(v[1], v[0]))

    @staticmethod
    def _wrap_to_pi(angle):
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
