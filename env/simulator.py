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

        self.gripper_joints = [
            self._get_joint_id("finger_joint1"),
            self._get_joint_id("finger_joint2"),
        ]
        self.gripper_qpos_addr = [
            self.model.jnt_qposadr[self.gripper_joints[0]],
            self.model.jnt_qposadr[self.gripper_joints[1]],
        ]

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

    def set_ee_position(self, target_pos):
        """
        Simple mocap-based EE positioning.
        """
        # Find mocap body (assumes panda_mocap exists)
        mocap_id = 0  # usually first mocap body
        self.data.mocap_pos[mocap_id] = target_pos
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
