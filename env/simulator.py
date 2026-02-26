import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path


class MujocoSimulator:
    """
    Minimal MuJoCo wrapper for Active Inference.

    Responsibilities:
    - load model
    - step simulation
    - expose limited, clean API
    """

    def __init__(
        self,
        model_path,
        render=True,
        record_path=None,
        record_fps=25,
        record_width=960,
        record_height=540,
        record_every_steps=1,
        record_camera="watching",
    ):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.render = render
        self.viewer = None
        self.renderer = None
        self.record_enabled = bool(record_path)
        self.record_path = Path(record_path).expanduser().resolve() if record_path else None
        self.record_fps = max(1, int(record_fps))
        self.record_width = max(64, int(record_width))
        self.record_height = max(64, int(record_height))
        self.record_every_steps = max(1, int(record_every_steps))
        self.record_camera = str(record_camera) if record_camera else None
        self.record_step_counter = 0
        self.record_frame_idx = 0
        self.record_writer = None
        self.record_frames_dir = None
        self.record_mode = "disabled"
        self._record_camera_fallback_used = False

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
        self.obj_joint_id = self._get_joint_id("obj_joint")
        self.obj_joint_qpos_addr = self.model.jnt_qposadr[self.obj_joint_id]
        self.obj_joint_dof_addr = self.model.jnt_dofadr[self.obj_joint_id]
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

        if self.record_enabled:
            self._init_recording()

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
            self._capture_record_frame()
            if self.viewer:
                self.viewer.sync()

    def _init_recording(self):
        if self.record_path is None:
            self.record_enabled = False
            return
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        if not hasattr(mujoco, "Renderer"):
            print("[Record] mujoco.Renderer is not available. Video recording disabled.")
            self.record_enabled = False
            return

        self.renderer = mujoco.Renderer(
            self.model,
            width=self.record_width,
            height=self.record_height,
        )

        try:
            import imageio.v2 as imageio

            writer_kwargs = {"fps": self.record_fps}
            if self.record_path.suffix.lower() == ".mp4":
                writer_kwargs["codec"] = "libx264"
            self.record_writer = imageio.get_writer(str(self.record_path), **writer_kwargs)
            self.record_mode = "video"
            print(
                f"[Record] video={self.record_path} fps={self.record_fps} "
                f"size={self.record_width}x{self.record_height} every={self.record_every_steps}step"
            )
        except Exception as exc:
            stem = self.record_path.stem if self.record_path.stem else "sim_record"
            self.record_frames_dir = self.record_path.parent / f"{stem}_frames"
            self.record_frames_dir.mkdir(parents=True, exist_ok=True)
            self.record_mode = "frames"
            print(
                f"[Record] imageio unavailable ({type(exc).__name__}). "
                f"Saving PNG frames to {self.record_frames_dir} instead."
            )

    def _capture_record_frame(self):
        if not self.record_enabled or self.renderer is None:
            return
        self.record_step_counter += 1
        if (self.record_step_counter - 1) % self.record_every_steps != 0:
            return

        try:
            if self.record_camera:
                self.renderer.update_scene(self.data, camera=self.record_camera)
            else:
                self.renderer.update_scene(self.data)
        except Exception:
            self.renderer.update_scene(self.data)
            if not self._record_camera_fallback_used:
                self._record_camera_fallback_used = True
                if self.record_camera:
                    print(
                        f"[Record] camera='{self.record_camera}' not found. "
                        "Falling back to default camera."
                    )
        frame = self.renderer.render()

        if self.record_mode == "video" and self.record_writer is not None:
            self.record_writer.append_data(frame)
        elif self.record_mode == "frames" and self.record_frames_dir is not None:
            try:
                from PIL import Image
            except Exception as exc:
                raise RuntimeError(
                    "Pillow is required for frame-sequence fallback recording."
                ) from exc
            out_path = self.record_frames_dir / f"frame_{self.record_frame_idx:06d}.png"
            Image.fromarray(frame).save(out_path)

        self.record_frame_idx += 1

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

    def close(self):
        if self.record_writer is not None:
            self.record_writer.close()
            self.record_writer = None
            print(
                f"[Record] saved {self.record_frame_idx} frame(s) to {self.record_path}"
            )
        elif self.record_mode == "frames" and self.record_frames_dir is not None:
            print(
                f"[Record] saved {self.record_frame_idx} PNG frame(s) to {self.record_frames_dir}"
            )

        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None and hasattr(self.viewer, "close"):
            self.viewer.close()
            self.viewer = None

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
            "sim_time": float(self.data.time),
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

    def set_object_pose(self, position, quat_wxyz=None):
        """
        Set free-joint object world pose.
        position: [x,y,z]
        quat_wxyz: [w,x,y,z] (optional, keep current if None)
        """
        pos = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(pos)):
            raise ValueError("Object position must contain finite values.")

        if quat_wxyz is None:
            quat = np.asarray(self.get_object_orientation_quat(), dtype=float).reshape(4)
        else:
            quat = np.asarray(quat_wxyz, dtype=float).reshape(4)
        if not np.all(np.isfinite(quat)):
            raise ValueError("Object quaternion must contain finite values.")
        qn = float(np.linalg.norm(quat))
        if qn <= 1e-9:
            raise ValueError("Object quaternion norm must be non-zero.")
        quat = quat / qn

        if int(self.model.jnt_type[self.obj_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError("Object joint 'obj_joint' is not a free joint.")

        qaddr = int(self.obj_joint_qpos_addr)
        daddr = int(self.obj_joint_dof_addr)
        self.data.qpos[qaddr : qaddr + 3] = pos
        self.data.qpos[qaddr + 3 : qaddr + 7] = quat
        self.data.qvel[daddr : daddr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_object_grasp_width_estimate(self):
        """
        Estimate object lateral grasp width from object body geoms.
        Returns conservative width in meters, or NaN when unavailable.
        """
        geom_start = int(self.model.body_geomadr[self.obj_body_id])
        geom_count = int(self.model.body_geomnum[self.obj_body_id])
        if geom_count <= 0:
            return float("nan")

        widths = []
        for gid in range(geom_start, geom_start + geom_count):
            gtype = int(self.model.geom_type[gid])
            gsize = np.asarray(self.model.geom_size[gid], dtype=float).reshape(-1)
            width = float("nan")

            if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                # Top-down pinch width from larger planar half-extent.
                width = 2.0 * float(max(gsize[0], gsize[1]))
            elif gtype in (int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
                width = 2.0 * float(gsize[0])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                width = 2.0 * float(gsize[0])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
                width = 2.0 * float(max(gsize[0], gsize[1]))
            else:
                # Fallback for mesh/other: use bounding sphere diameter.
                rbound = float(self.model.geom_rbound[gid])
                if np.isfinite(rbound) and rbound > 0.0:
                    width = 2.0 * rbound

            if np.isfinite(width) and width > 0.0:
                widths.append(float(width))

        if not widths:
            return float("nan")
        return float(max(widths))

    # ------------------------------------------------
    # Target (place site)
    # ------------------------------------------------
    def get_target_position(self):
        return self.data.site_xpos[self.target_site_id].copy()

    def set_target_position(self, target_pos):
        """
        Set world-frame position of the target site marker.
        This updates the red target dot used by sensors/logs.
        """
        target_world = np.asarray(target_pos, dtype=float).reshape(3)
        if not np.all(np.isfinite(target_world)):
            raise ValueError("Target position must contain finite values.")

        site_id = int(self.target_site_id)
        body_id = int(self.model.site_bodyid[site_id])
        body_world_pos = np.asarray(self.data.xpos[body_id], dtype=float).reshape(3)
        body_world_rot = np.asarray(self.data.xmat[body_id], dtype=float).reshape(3, 3)
        local_pos = body_world_rot.T @ (target_world - body_world_pos)
        self.model.site_pos[site_id] = np.asarray(local_pos, dtype=float)
        mujoco.mj_forward(self.model, self.data)

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
