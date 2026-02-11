import mujoco
import mujoco.viewer
import numpy as np
import time


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
        self.mocap_id = self._get_mocap_id("panda_mocap")

        self.gripper_joints = [
            self._get_joint_id("finger_joint1"),
            self._get_joint_id("finger_joint2"),
        ]
        self.gripper_qpos_addr = [
            self.model.jnt_qposadr[self.gripper_joints[0]],
            self.model.jnt_qposadr[self.gripper_joints[1]],
        ]
        self.mocap_tracking_gain = 0.35
        self.ee_feedback_gain = 0.20
        self.max_mocap_step = 0.03

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

    def _get_mocap_id(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        mocap_id = int(self.model.body_mocapid[body_id])
        if mocap_id < 0:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        return mocap_id

    # ------------------------------------------------
    # Simulation stepping
    # ------------------------------------------------
    def step(self, steps=1):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if self.viewer:
                self.viewer.sync()

    # ------------------------------------------------
    # State access (GROUND TRUTH — internal only)
    # ------------------------------------------------
    def get_state(self):
        """
        Return true simulator state.
        ONLY sensors may use this.
        """

        ee_pos = self.get_ee_position()
        obj_pos = self.get_object_position()
        gripper_width = self.get_gripper_width()

        return {
            "ee_pos": ee_pos,
            "obj_pos": obj_pos,
            "gripper_width": gripper_width,
        }

    # ------------------------------------------------
    # End-effector
    # ------------------------------------------------
    def get_ee_position(self):
        return self.data.site_xpos[self.ee_site_id].copy()

    def get_mocap_position(self):
        return self.data.mocap_pos[self.mocap_id].copy()

    def set_ee_position(self, target_pos):
        """
        Simple mocap-based EE positioning.
        """
        target_pos = np.array(target_pos, dtype=float)
        current_mocap = self.get_mocap_position()
        current_ee = self.get_ee_position()

        # Blend mocap-target tracking with EE feedback so weld lag does not invert behavior.
        mocap_err = target_pos - current_mocap
        ee_err = target_pos - current_ee
        step = self.mocap_tracking_gain * mocap_err + self.ee_feedback_gain * ee_err

        step_norm = np.linalg.norm(step)
        if step_norm > self.max_mocap_step:
            step = (step / step_norm) * self.max_mocap_step

        self.data.mocap_pos[self.mocap_id] = current_mocap + step
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------
    # Object
    # ------------------------------------------------
    def get_object_position(self):
        return self.data.site_xpos[self.obj_site_id].copy()

    # ------------------------------------------------
    # Gripper
    # ------------------------------------------------
    def get_gripper_width(self):
        q1 = self.data.qpos[self.gripper_qpos_addr[0]]
        q2 = self.data.qpos[self.gripper_qpos_addr[1]]
        return q1 + q2

    def open_gripper(self):
        self._set_gripper_width(0.04)

    def close_gripper(self):
        self._set_gripper_width(0.0)

    def _set_gripper_width(self, width):
        half = width / 2.0
        self.data.qpos[self.gripper_qpos_addr[0]] = half
        self.data.qpos[self.gripper_qpos_addr[1]] = half
        mujoco.mj_forward(self.model, self.data)
