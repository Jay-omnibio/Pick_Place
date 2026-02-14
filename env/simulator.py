import mujoco
import mujoco.viewer
import numpy as np


class MujocoSimulator:
    """
    Minimal MuJoCo wrapper for Active Inference.

    Responsibilities:
    - load model
    - step simulation
    - expose limited, clean API
    """

    def __init__(self, model_path, render=True):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render = render
        self.viewer = None

        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Cache IDs
        self.ee_site_id = self._get_site_id("ee_center_site")
        self.obj_site_id = self._get_site_id("obj_site")
        self.target_site_id = self._get_site_id("target")
        self.mocap_id = self._get_mocap_id("panda_mocap")
        self.obj_body_id = self._get_body_id("obj")
        self.hand_body_id = self._get_body_id("hand")
        self.gripper_body_ids = {
            self._get_body_id("hand"),
            self._get_body_id("left_finger"),
            self._get_body_id("right_finger"),
        }

        self.gripper_joints = [
            self._get_joint_id("finger_joint1"),
            self._get_joint_id("finger_joint2"),
        ]
        self.joint7_id = self._get_joint_id("joint7")
        self.joint7_qpos_addr = self.model.jnt_qposadr[self.joint7_id]
        self.gripper_qpos_addr = [
            self.model.jnt_qposadr[self.gripper_joints[0]],
            self.model.jnt_qposadr[self.gripper_joints[1]],
        ]
        self.gripper_dof_addr = [
            self.model.jnt_dofadr[self.gripper_joints[0]],
            self.model.jnt_dofadr[self.gripper_joints[1]],
        ]

        # Prefer actuator-driven gripper commands when available.
        aid_r = self._get_actuator_id("r_gripper_finger_joint")
        aid_l = self._get_actuator_id("l_gripper_finger_joint")
        self.gripper_actuator_ids = [aid_r, aid_l]
        self.use_gripper_actuators = (aid_r >= 0 and aid_l >= 0)

        # Conservative tracking parameters to avoid dynamic blowups.
        self.mocap_tracking_gain = 0.20
        self.ee_feedback_gain = 0.10
        self.max_mocap_step = 0.008
        self.max_ee_feedback_err = 0.12
        # Keep Z floor low enough so Descend can reach near-object grasp height.
        self.workspace_min = np.array([0.05, -0.50, 0.005], dtype=float)
        self.workspace_max = np.array([0.85, 0.50, 0.95], dtype=float)

        # Initial settle
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

    # ------------------------------------------------
    # Utility ID helpers
    # ------------------------------------------------
    def _get_site_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)

    def _get_joint_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    def _get_actuator_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)

    def _get_mocap_id(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        return mocap_id

    def _get_body_id(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    # ------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------
    def step(self, steps=1):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if not self._is_dynamics_finite():
                self._recover_from_instability()
            if self.viewer:
                self.viewer.sync()

    def _is_dynamics_finite(self):
        return (
            np.all(np.isfinite(self.data.qpos))
            and np.all(np.isfinite(self.data.qvel))
            and np.all(np.isfinite(self.data.qacc))
        )

    def _recover_from_instability(self):
        """
        Best-effort recovery when MuJoCo reports unstable dynamics.
        Keeps current pose, zeroes velocities, and re-anchors mocap to EE.
        """
        if not np.all(np.isfinite(self.data.qpos)):
            mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qacc_warmstart[:] = 0.0
        ee_pos = self.get_ee_position()
        self.data.mocap_pos[self.mocap_id] = np.clip(ee_pos, self.workspace_min, self.workspace_max)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------
    # State access (GROUND TRUTH - internal only)
    # ------------------------------------------------
    def get_state(self):
        """
        Return true simulator state.
        ONLY sensors may use this.
        """

        ee_pos = self.get_ee_position()
        obj_pos = self.get_object_position()
        target_pos = self.get_target_position()
        gripper_width = self.get_gripper_width()
        obj_gripper_contact = self.get_object_gripper_contact()

        return {
            "ee_pos": ee_pos,
            "obj_pos": obj_pos,
            "obj_quat_wxyz": self.get_object_orientation_quat(),
            "target_pos": target_pos,
            "gripper_width": gripper_width,
            "gripper_speed": self.get_gripper_speed(),
            "obj_gripper_contact": obj_gripper_contact,
            "joint7_pos": self.get_joint7_position(),
            "hand_quat_wxyz": self.get_hand_orientation_quat(),
        }

    # ------------------------------------------------
    # End-effector
    # ------------------------------------------------
    def get_ee_position(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_mocap_position(self):
        return self.data.mocap_pos[self.mocap_id].copy()

    def get_hand_orientation_quat(self):
        # World orientation of hand body as quaternion [w, x, y, z].
        return self.data.xquat[self.hand_body_id].copy()

    def get_joint7_position(self):
        return float(self.data.qpos[self.joint7_qpos_addr])

    def set_ee_position(self, target_pos):
        """
        Simple mocap-based EE positioning.
        """
        target_pos = np.array(target_pos, dtype=float)
        target_pos = np.clip(target_pos, self.workspace_min, self.workspace_max)
        current_mocap = self.get_mocap_position()
        current_ee = self.get_ee_position()

        # Blend mocap-target tracking with EE feedback so weld lag does not invert behavior.
        mocap_err = target_pos - current_mocap
        ee_err = target_pos - current_ee
        ee_err_norm = np.linalg.norm(ee_err)
        if ee_err_norm > self.max_ee_feedback_err and ee_err_norm > 0:
            ee_err = (ee_err / ee_err_norm) * self.max_ee_feedback_err
        step = self.mocap_tracking_gain * mocap_err + self.ee_feedback_gain * ee_err

        step_norm = np.linalg.norm(step)
        if step_norm > self.max_mocap_step:
            step = (step / step_norm) * self.max_mocap_step

        next_mocap = np.clip(current_mocap + step, self.workspace_min, self.workspace_max)
        self.data.mocap_pos[self.mocap_id] = next_mocap
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------
    # Object
    # ------------------------------------------------
    def get_object_position(self):
        return self.data.site_xpos[self.obj_site_id].copy()

    def get_object_orientation_quat(self):
        # World orientation of object body as quaternion [w, x, y, z].
        return self.data.xquat[self.obj_body_id].copy()

    # ------------------------------------------------
    # Target (place site)
    # ------------------------------------------------
    def get_target_position(self):
        return self.data.site_xpos[self.target_site_id].copy()

    def get_object_gripper_contact(self):
        """
        Return 1 if object has physical contact with hand/finger bodies.
        """
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if (b1 == self.obj_body_id and b2 in self.gripper_body_ids) or (
                b2 == self.obj_body_id and b1 in self.gripper_body_ids
            ):
                return 1
        return 0

    # ------------------------------------------------
    # Gripper
    # ------------------------------------------------
    def get_gripper_width(self):
        q1 = self.data.qpos[self.gripper_qpos_addr[0]]
        q2 = self.data.qpos[self.gripper_qpos_addr[1]]
        return q1 + q2

    def get_gripper_speed(self):
        v1 = self.data.qvel[self.gripper_dof_addr[0]]
        v2 = self.data.qvel[self.gripper_dof_addr[1]]
        return v1 + v2

    def open_gripper(self):
        self.set_gripper_width(0.04)

    def close_gripper(self):
        self.set_gripper_width(0.0)

    def set_gripper_width(self, width):
        # width is total gap across both fingers.
        width = float(np.clip(width, 0.0, 0.08))
        self._set_gripper_width(width)

    def _set_gripper_width(self, width):
        half = width / 2.0

        if self.use_gripper_actuators:
            # Command both finger actuators to target joint position.
            for aid in self.gripper_actuator_ids:
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[aid]
                self.data.ctrl[aid] = float(np.clip(half, ctrl_min, ctrl_max))
            return

        # Fallback for models without named finger actuators.
        self.data.qpos[self.gripper_qpos_addr[0]] = half
        self.data.qpos[self.gripper_qpos_addr[1]] = half
        mujoco.mj_forward(self.model, self.data)
