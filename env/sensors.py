import numpy as np


class SensorSuite:
    """
    Sensor abstraction layer.

    This class:
    - reads TRUE simulator state
    - adds noise & partial observability
    - returns observations ONLY (no ground truth)

    Active Inference MUST consume only this output.
    """

    def __init__(self, config):
        self.ee_noise_std = config["ee_sensor"]["noise_std"]
        self.obj_noise_std = config["object_sensor"]["noise_std"]
        self.obj_yaw_noise_std = config["object_sensor"].get("yaw_noise_std", 0.0)
        self.target_noise_std = config.get("target_sensor", {}).get(
            "noise_std", config["object_sensor"]["noise_std"]
        )
        self.contact_threshold = config["contact_sensor"]["distance_threshold"]

    # -------------------------------
    # Main interface
    # -------------------------------
    def get_observation(self, sim_state):
        """
        sim_state: dict with true simulator values
        (MuJoCo wrapper provides this)

        returns: observation dict o_t
        """

        ee_obs = self._sense_ee(sim_state)
        obj_obs = self._sense_object_relative(sim_state)
        target_obs = self._sense_target_relative(sim_state)
        obj_yaw_obs = self._sense_object_yaw(sim_state)
        grip_obs = self._sense_gripper(sim_state)
        contact_obs = self._sense_contact(sim_state)

        return {
            "o_ee": ee_obs,
            "o_obj": obj_obs,
            "o_target": target_obs,
            "o_obj_yaw": obj_yaw_obs,
            "o_grip": grip_obs,
            "o_contact": contact_obs,
        }

    # -------------------------------
    # Individual sensors
    # -------------------------------
    def _sense_ee(self, sim_state):
        """
        Proprioceptive EE sensor
        High precision, small noise
        """
        true_ee = sim_state["ee_pos"]
        noise = np.random.normal(0, self.ee_noise_std, size=3)
        return true_ee + noise

    def _sense_object_relative(self, sim_state):
        """
        Vision-like object sensor
        Object position RELATIVE to EE
        Lower precision
        """
        true_obj = sim_state["obj_pos"]
        true_ee = sim_state["ee_pos"]

        relative_pos = true_obj - true_ee
        noise = np.random.normal(0, self.obj_noise_std, size=3)
        return relative_pos + noise

    def _sense_target_relative(self, sim_state):
        """
        Place-target observation (relative to EE).
        Treated like a vision-style observation (moderate noise).
        """
        true_target = sim_state["target_pos"]
        true_ee = sim_state["ee_pos"]
        relative_pos = true_target - true_ee
        noise = np.random.normal(0, self.target_noise_std, size=3)
        return relative_pos + noise

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz):
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    def _sense_object_yaw(self, sim_state):
        """
        Object yaw around world Z (radians), from true object quaternion.
        """
        quat = sim_state.get("obj_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        yaw = self._quat_wxyz_to_yaw(quat)
        if self.obj_yaw_noise_std > 0.0:
            yaw += float(np.random.normal(0.0, self.obj_yaw_noise_std))
        return float(yaw)

    def _sense_gripper(self, sim_state):
        """
        Gripper encoder
        High precision
        """
        return sim_state["gripper_width"]

    def _sense_contact(self, sim_state):
        """
        Binary contact sensor
        Probabilistic / threshold-based
        """
        if "obj_gripper_contact" in sim_state:
            return int(sim_state["obj_gripper_contact"])

        dist = np.linalg.norm(sim_state["obj_pos"] - sim_state["ee_pos"])

        if dist < self.contact_threshold:
            return 1
        else:
            return 0
