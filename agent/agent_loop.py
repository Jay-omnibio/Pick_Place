import csv
import os
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from policies.scripted_pick_place import ScriptedPickPlacePolicy
from tasks.pick_place_fsm import Phase, TaskConfig, step_fsm
from inference_interface import infer_beliefs
from inference.action_selection import select_action
from agent.ai_behavior_tree import AIPickPlaceBehaviorTree


class ActiveInferenceAgent:
    """
    Baseline pick-and-place agent loop:
    sense -> task_fsm -> policy -> act

    Uses sensor_backend and actuator_backend for sim/real swap.
    """

    @staticmethod
    def _env_flag(name, default=False):
        val = os.getenv(name)
        if val is None:
            return bool(default)
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    def __init__(
        self,
        simulator,
        sensor_backend,
        actuator_backend,
        control_mode="fsm",
        log_every_steps=100,
        logs_dir="logs",
        task_cfg=None,
        policy_cfg=None,
        active_inference_cfg=None,
    ):
        self.simulator = simulator
        self.sensor_backend = sensor_backend
        self.actuator_backend = actuator_backend
        self.control_mode = str(control_mode).lower()
        if self.control_mode not in ("fsm", "active_inference"):
            raise ValueError(f"Unsupported control_mode: {control_mode}")
        self.log_every_steps = max(1, int(log_every_steps))
        self.step_count = 0
        if task_cfg is None:
            raise ValueError("task_cfg is required. Load it from config/fsm_config.yaml.")
        if policy_cfg is None:
            raise ValueError("policy_cfg is required. Load it from config/fsm_config.yaml.")
        if self.control_mode == "active_inference" and active_inference_cfg is None:
            raise ValueError(
                "active_inference_cfg is required in active_inference mode. "
                "Load it from config/active_inference_config.yaml."
            )
        self.task_cfg = task_cfg
        self.active_inference_cfg = dict(active_inference_cfg or {})
        self.task_state = None
        self.policy = ScriptedPickPlacePolicy(task_cfg=self.task_cfg, policy_cfg=policy_cfg)
        self.current_belief = None
        self.ai_startup_settle_steps = max(0, int(os.getenv("AI_STARTUP_SETTLE_STEPS", "80")))
        self.active_inference_params = (
            self._build_active_inference_params() if self.control_mode == "active_inference" else {}
        )
        self.ai_bt = (
            AIPickPlaceBehaviorTree(
                max_retries=int(self.active_inference_params.get("max_retries", 3)),
                reach_reentry_cooldown_steps=int(
                    self.active_inference_params.get("reach_reentry_cooldown_steps", 20)
                ),
                progress_eps=float(self.active_inference_params.get("reach_progress_eps", 1e-3)),
                no_progress_limit=int(self.active_inference_params.get("reach_stall_steps", 240)),
                set_priors_enabled=bool(self.active_inference_params.get("bt_set_priors_enabled", True)),
                retry_reach_z_step=float(self.active_inference_params.get("bt_retry_reach_z_step", 0.005)),
                retry_reach_z_max=float(self.active_inference_params.get("bt_retry_reach_z_max", 0.02)),
                vfe_recover_enabled=bool(self.active_inference_params.get("vfe_recover_enabled", False)),
                vfe_recover_threshold=float(self.active_inference_params.get("vfe_recover_threshold", 2.0)),
                vfe_recover_steps=int(self.active_inference_params.get("vfe_recover_steps", 50)),
                branch_retry_cap=int(self.active_inference_params.get("bt_branch_retry_cap", 2)),
                global_recovery_cap=int(self.active_inference_params.get("bt_global_recovery_cap", 6)),
                rescan_hold_steps=int(self.active_inference_params.get("bt_rescan_hold_steps", 40)),
                reapproach_offset_xy=float(self.active_inference_params.get("bt_reapproach_offset_xy", 0.01)),
                safe_backoff_hold_steps=int(self.active_inference_params.get("bt_safe_backoff_hold_steps", 30)),
                safe_backoff_z_boost=float(self.active_inference_params.get("bt_safe_backoff_z_boost", 0.02)),
            )
            if self.control_mode == "active_inference"
            else None
        )
        self.ai_risk_detection_enabled = bool(
            self.active_inference_params.get("risk_detection_enabled", False)
        )
        self.ai_singularity_dq_ratio_threshold = float(
            self.active_inference_params.get("singularity_dq_ratio_threshold", 5.0)
        )
        self.ai_singularity_no_progress_steps = int(
            self.active_inference_params.get("singularity_no_progress_steps", 30)
        )
        self.ai_unintended_contact_warn_steps = int(
            self.active_inference_params.get("unintended_contact_warn_steps", 12)
        )
        self.ai_risk_progress_eps = float(self.active_inference_params.get("reach_progress_eps", 0.002))
        self.ai_prev_risk_phase = ""
        self.ai_prev_phase_error = None
        self.ai_phase_no_progress_steps = 0
        self.ai_singularity_counter = 0
        self.ai_unintended_contact_counter = 0
        self.ai_risk_state = {
            "dq_ratio": 0.0,
            "phase_no_progress_steps": 0,
            "singularity_counter": 0,
            "singularity_warn": 0,
            "unexpected_contact": 0,
            "unintended_contact_counter": 0,
            "unintended_contact_warn": 0,
        }
        # Runtime timing/freshness monitoring (P0.1 robotics observability).
        self.loop_target_hz = float(os.getenv("LOOP_TARGET_HZ", "50.0"))
        self.loop_target_ms = (
            (1000.0 / self.loop_target_hz) if self.loop_target_hz > 0.0 else 20.0
        )
        self.loop_dt_warn_ms = float(os.getenv("LOOP_DT_WARN_MS", "40.0"))
        self.obs_age_warn_ms = float(os.getenv("OBS_AGE_WARN_MS", "120.0"))
        self._last_step_wall_time = None
        self.loop_dt_ms = 0.0
        self.loop_dt_ema_ms = 0.0
        self.loop_jitter_ms = 0.0
        self.loop_dt_max_ms = 0.0
        self.loop_overrun_count = 0
        self.obs_timestamp = 0.0
        self.obs_age_ms = 0.0
        self.obs_stale_warn = 0
        self._prev_obs_timestamp = None
        self._prev_obs_stale_warn = 0
        self._ai_settle_logged = False
        self.prev_phase = None
        self.prev_contact = None
        self.prev_escape_active = 0
        self.prev_ai_release_warning = 0
        self.prev_ai_singularity_warn = 0
        self.prev_ai_unintended_contact_warn = 0
        self.log_contact_events = self._env_flag("LOG_CONTACT_EVENTS", False)
        self.log_pose_debug = self._env_flag("LOG_POSE_DEBUG", False)
        self.pause_on_reach_to_descend = self._env_flag("PAUSE_ON_REACH_TO_DESCEND", True)
        self.pause_on_reach_to_descend_once = self._env_flag("PAUSE_ON_REACH_TO_DESCEND_ONCE", True)
        self.pause_on_phase_change = self._env_flag("PAUSE_ON_PHASE_CHANGE", True)
        self.pause_on_grip_start = self._env_flag("PAUSE_ON_GRIP_START", True)
        self.pause_pending = False
        self.reach_to_descend_pause_already_triggered = False
        self.pause_reason = ""
        self.last_descend_gate = None
        self.last_transition_reason = ""
        self.prev_action_grip = None
        self._hb_prev_metrics = None

        self.obs_history = deque(maxlen=50)
        self.action_history = deque(maxlen=50)
        self.ee_true_history = deque(maxlen=120)
        self.escape_cooldown = 0
        self.escape_active = 0
        self.stall_window = 120
        self.stall_disp_threshold = 0.0015
        self.escape_steps = 10
        self.recovery_steps = 24
        self.recovery_steps_remaining = 0
        self.recovery_height = 0.10
        self.recovery_max_step = 0.010
        # In active-inference mode, BT should own recovery by default.
        self.ai_allow_lowlevel_escape = self._env_flag("AI_ALLOW_LOWLEVEL_ESCAPE", False)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path = self.logs_dir / f"run_{self.run_id}.csv"
        self._csv_file = self.log_csv_path.open("w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_columns())
        self._csv_writer.writeheader()

    def _csv_columns(self):
        return [
            "step",
            "phase",
            "sim_time",
            "obs_timestamp",
            "obs_age_ms",
            "obs_stale_warn",
            "loop_dt_ms",
            "loop_dt_ema_ms",
            "loop_jitter_ms",
            "loop_target_ms",
            "loop_overrun_count",
            "obs_dt_s",
            "est_ee_vel_norm",
            "est_obj_vel_norm",
            "est_target_vel_norm",
            "distance_to_object",
            "reach_error",
            "pregrasp_error",
            "descend_error",
            "descend_x_error",
            "descend_y_error",
            "descend_z_error",
            "true_reach_xy_error",
            "true_descend_xy_error",
            "obs_reach_xy_error",
            "obs_descend_xy_error",
            "obs_preplace_error",
            "obs_place_error",
            "active_reach_ref_x",
            "active_reach_ref_y",
            "active_reach_ref_z",
            "active_descend_ref_x",
            "active_descend_ref_y",
            "active_descend_ref_z",
            "phase_step",
            "distance_to_target",
            "s_ee_x",
            "s_ee_y",
            "s_ee_z",
            "s_obj_rel_x",
            "s_obj_rel_y",
            "s_obj_rel_z",
            "s_target_rel_x",
            "s_target_rel_y",
            "s_target_rel_z",
            "ai_belief_ee_x",
            "ai_belief_ee_y",
            "ai_belief_ee_z",
            "ai_belief_obj_rel_x",
            "ai_belief_obj_rel_y",
            "ai_belief_obj_rel_z",
            "ai_belief_target_rel_x",
            "ai_belief_target_rel_y",
            "ai_belief_target_rel_z",
            "ai_belief_obj_yaw",
            "action_move_x",
            "action_move_y",
            "action_move_z",
            "action_grip",
            "escape_active",
            "obs_contact",
            "obs_grip",
            "true_ee_x",
            "true_ee_y",
            "true_ee_z",
            "true_obj_x",
            "true_obj_y",
            "true_obj_z",
            "true_target_x",
            "true_target_y",
            "true_target_z",
            "true_joint7_pos",
            "true_hand_roll",
            "true_hand_pitch",
            "true_hand_yaw",
            "ai_obs_confidence",
            "ai_vfe_total",
            "ai_phase_conf_ok",
            "ai_phase_vfe_ok",
            "ai_phase_gate_ok",
            "ai_bt_status",
            "ai_bt_reason",
            "ai_retry_reason",
            "ai_failure_reason",
            "ai_recovery_branch",
            "ai_recovery_branch_retry",
            "ai_recovery_global_count",
            "ai_release_detach_counter",
            "ai_release_stable_counter",
            "ai_release_reapproach_count",
        ]

    @staticmethod
    def _quat_wxyz_to_rpy(quat_wxyz):
        w, x, y, z = [float(v) for v in quat_wxyz]
        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz):
        w, x, y, z = [float(v) for v in quat_wxyz]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(math.atan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _obj_rel_world_to_local(obj_rel_world, obj_yaw):
        rel = np.asarray(obj_rel_world, dtype=float).copy()
        c = float(np.cos(float(obj_yaw)))
        s = float(np.sin(float(obj_yaw)))
        x_w, y_w = float(rel[0]), float(rel[1])
        # local = Rz(-yaw) * world
        rel[0] = c * x_w + s * y_w
        rel[1] = -s * x_w + c * y_w
        return rel

    def _obj_rel_for_reach_descend(self, obj_rel_world, obj_yaw):
        rel = np.asarray(obj_rel_world, dtype=float)
        if getattr(self.task_cfg, "use_object_local_xy_errors", True):
            return self._obj_rel_world_to_local(rel, obj_yaw)
        return rel.copy()

    @staticmethod
    def _obs_obj_yaw(observation):
        raw = observation.get("o_obj_yaw", 0.0)
        try:
            yaw = float(np.asarray(raw, dtype=float).reshape(-1)[0])
            if np.isfinite(yaw):
                return yaw
        except (TypeError, ValueError, IndexError):
            pass
        return 0.0

    def _true_obj_yaw(self, sim_state):
        quat = sim_state.get("obj_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        return self._quat_wxyz_to_yaw(quat)

    def _phase_name(self):
        if self.control_mode == "active_inference":
            if self.current_belief is None:
                return "Reach"
            return str(self.current_belief.get("phase", "Reach"))
        return self.task_state.phase.value if self.task_state is not None else Phase.ReachAbove.value

    def is_terminal(self):
        if self.control_mode == "active_inference":
            if self.current_belief is None:
                return False
            phase = str(self.current_belief.get("phase", "Reach"))
            return phase in ("Done", "Failure")
        if self.task_state is None:
            return False
        phase = Phase(self.task_state.phase)
        return phase in (Phase.Done, Phase.Failure)

    def terminal_phase(self):
        if self.control_mode == "active_inference":
            if self.current_belief is None:
                return ""
            return str(self.current_belief.get("phase", ""))
        if self.task_state is None:
            return ""
        return Phase(self.task_state.phase).value

    def _reset_ai_risk_state(self):
        self.ai_prev_risk_phase = ""
        self.ai_prev_phase_error = None
        self.ai_phase_no_progress_steps = 0
        self.ai_singularity_counter = 0
        self.ai_unintended_contact_counter = 0
        self.ai_risk_state = {
            "dq_ratio": 0.0,
            "phase_no_progress_steps": 0,
            "singularity_counter": 0,
            "singularity_warn": 0,
            "unexpected_contact": 0,
            "unintended_contact_counter": 0,
            "unintended_contact_warn": 0,
        }

    @staticmethod
    def _safe_norm(vec):
        n = float(np.linalg.norm(np.asarray(vec, dtype=float)))
        if not np.isfinite(n):
            return float("inf")
        return n

    @staticmethod
    def _safe_float_scalar(value, default=0.0):
        try:
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.size > 0 and np.isfinite(arr[0]):
                return float(arr[0])
        except (TypeError, ValueError):
            pass
        return float(default)

    def _update_runtime_timing(self):
        now = time.perf_counter()
        if self._last_step_wall_time is None:
            self.loop_dt_ms = self.loop_target_ms
            self.loop_dt_ema_ms = self.loop_dt_ms
            self.loop_jitter_ms = 0.0
        else:
            dt_ms = max(0.0, (now - self._last_step_wall_time) * 1000.0)
            self.loop_dt_ms = dt_ms
            self.loop_jitter_ms = abs(dt_ms - self.loop_target_ms)
            alpha = 0.15
            self.loop_dt_ema_ms = (1.0 - alpha) * self.loop_dt_ema_ms + alpha * dt_ms
            if dt_ms > self.loop_dt_max_ms:
                self.loop_dt_max_ms = dt_ms
            if dt_ms > self.loop_dt_warn_ms:
                self.loop_overrun_count += 1
        self._last_step_wall_time = now

    def _update_observation_freshness(self, sim_state, observation):
        sim_time = self._safe_float_scalar(sim_state.get("sim_time", 0.0), default=0.0)
        obs_ts = self._safe_float_scalar(observation.get("o_timestamp", sim_time), default=sim_time)
        self.obs_timestamp = obs_ts
        age_s = sim_time - obs_ts
        if not np.isfinite(age_s):
            age_s = 0.0
        # Do not report negative age from minor timestamp order noise.
        age_s = max(0.0, age_s)
        self.obs_age_ms = age_s * 1000.0
        self.obs_stale_warn = int(self.obs_age_ms > self.obs_age_warn_ms)

        if self._prev_obs_timestamp is not None and obs_ts < self._prev_obs_timestamp - 1e-9:
            print(
                f"[ObsWarn] step={self.step_count} non_monotonic_ts "
                f"prev={self._prev_obs_timestamp:.6f} now={obs_ts:.6f}"
            )
        if self.obs_stale_warn == 1 and self._prev_obs_stale_warn == 0:
            print(
                f"[ObsWarn] step={self.step_count} stale_observation "
                f"obs_age_ms={self.obs_age_ms:.1f} threshold_ms={self.obs_age_warn_ms:.1f}"
            )
        self._prev_obs_timestamp = obs_ts
        self._prev_obs_stale_warn = self.obs_stale_warn

    def _ai_phase_error(self):
        if self.current_belief is None:
            return None
        b = self.current_belief
        phase = str(b.get("phase", "Reach"))
        s_obj = np.asarray(b.get("s_obj_mean", [0.0, 0.0, 0.0]), dtype=float)
        s_target = np.asarray(b.get("s_target_mean", [0.0, 0.0, 0.0]), dtype=float)
        s_ee = np.asarray(b.get("s_ee_mean", [0.0, 0.0, 0.0]), dtype=float)

        if phase == "Reach":
            return self._safe_norm(s_obj - np.asarray(b.get("reach_obj_rel", [0.0, 0.0, 0.0]), dtype=float))
        if phase in ("Align", "PreGraspHold"):
            return self._safe_norm(s_obj - np.asarray(b.get("align_obj_rel", [0.0, 0.0, 0.0]), dtype=float))
        if phase in ("Descend", "CloseHold", "LiftTest"):
            return self._safe_norm(s_obj - np.asarray(b.get("descend_obj_rel", [0.0, 0.0, 0.0]), dtype=float))
        if phase == "MoveToPlaceAbove":
            return self._safe_norm(
                s_target - np.asarray(b.get("preplace_target_rel", [0.0, 0.0, 0.0]), dtype=float)
            )
        if phase == "DescendToPlace":
            return self._safe_norm(
                s_target - np.asarray(b.get("place_target_rel", [0.0, 0.0, 0.0]), dtype=float)
            )
        if phase == "Transit":
            transit_target = float(self.active_inference_params.get("transit_height", 0.35))
            return abs(transit_target - float(s_ee[2]))
        return None

    def _update_ai_risk_detection(self, observation):
        if (
            self.control_mode != "active_inference"
            or self.current_belief is None
            or not self.ai_risk_detection_enabled
        ):
            self._reset_ai_risk_state()
            return

        phase = str(self.current_belief.get("phase", "Reach"))
        phase_error = self._ai_phase_error()
        if phase != self.ai_prev_risk_phase:
            self.ai_prev_risk_phase = phase
            self.ai_prev_phase_error = phase_error
            self.ai_phase_no_progress_steps = 0
        elif phase_error is None or not np.isfinite(float(phase_error)):
            self.ai_prev_phase_error = phase_error
            self.ai_phase_no_progress_steps = 0
        else:
            prev_err = self.ai_prev_phase_error
            if prev_err is None or not np.isfinite(float(prev_err)):
                self.ai_phase_no_progress_steps = 0
            elif float(phase_error) < (float(prev_err) - self.ai_risk_progress_eps):
                self.ai_phase_no_progress_steps = 0
            else:
                self.ai_phase_no_progress_steps += 1
            self.ai_prev_phase_error = float(phase_error)

        controller = getattr(self.actuator_backend, "controller", None)
        if controller is not None:
            dq_raw = float(getattr(controller, "last_dq_norm_raw", 0.0))
            dq_applied = float(getattr(controller, "last_dq_norm_applied", 0.0))
        else:
            dq_raw = 0.0
            dq_applied = 0.0
        dq_ratio = dq_raw / max(dq_applied, 1e-6)
        singularity_like = (
            dq_ratio >= self.ai_singularity_dq_ratio_threshold
            and self.ai_phase_no_progress_steps > 0
        )
        if singularity_like:
            self.ai_singularity_counter += 1
        else:
            self.ai_singularity_counter = 0
        singularity_warn = int(self.ai_singularity_counter >= self.ai_singularity_no_progress_steps)

        contact = int(observation.get("o_contact", 0))
        expected_contact = phase in (
            "CloseHold",
            "LiftTest",
            "Transit",
            "MoveToPlaceAbove",
            "DescendToPlace",
            "Open",
            "Retreat",
            "Done",
        )
        if phase == "Descend":
            desc_vec = np.asarray(self.current_belief.get("s_obj_mean", [0.0, 0.0, 0.0]), dtype=float) - np.asarray(
                self.current_belief.get("descend_obj_rel", [0.0, 0.0, 0.0]), dtype=float
            )
            desc_z_err = abs(float(desc_vec[2]))
            desc_z_thr = float(self.active_inference_params.get("descend_z_threshold", 0.01))
            expected_contact = desc_z_err <= (2.0 * desc_z_thr)

        unexpected_contact = int(contact == 1 and not expected_contact)
        if unexpected_contact:
            self.ai_unintended_contact_counter += 1
        else:
            self.ai_unintended_contact_counter = 0
        unintended_contact_warn = int(
            self.ai_unintended_contact_counter >= self.ai_unintended_contact_warn_steps
        )

        self.ai_risk_state = {
            "dq_ratio": float(dq_ratio),
            "phase_no_progress_steps": int(self.ai_phase_no_progress_steps),
            "singularity_counter": int(self.ai_singularity_counter),
            "singularity_warn": int(singularity_warn),
            "unexpected_contact": int(unexpected_contact),
            "unintended_contact_counter": int(self.ai_unintended_contact_counter),
            "unintended_contact_warn": int(unintended_contact_warn),
        }

    def _get_references(self):
        if self.control_mode == "active_inference":
            if "reach_obj_rel" not in self.active_inference_params:
                raise KeyError("Missing active-inference config key: reach_obj_rel")
            if "descend_obj_rel" not in self.active_inference_params:
                raise KeyError("Missing active-inference config key: descend_obj_rel")
            if self.current_belief is not None:
                reach_ref = np.asarray(
                    self.current_belief.get("reach_obj_rel", self.active_inference_params["reach_obj_rel"]),
                    dtype=float,
                )
                descend_ref = np.asarray(
                    self.current_belief.get("descend_obj_rel", self.active_inference_params["descend_obj_rel"]),
                    dtype=float,
                )
            else:
                reach_ref = np.asarray(self.active_inference_params["reach_obj_rel"], dtype=float)
                descend_ref = np.asarray(self.active_inference_params["descend_obj_rel"], dtype=float)
            return reach_ref, descend_ref
        return self.task_cfg.pregrasp_obj_rel, self.task_cfg.grasp_obj_rel

    def _build_active_inference_params(self):
        """
        Build strict active-inference params from active_inference_config.yaml and
        controller values from common_robot.yaml.
        """
        out = dict(self.active_inference_cfg)
        controller = getattr(self.actuator_backend, "controller", None)
        if controller is None:
            raise ValueError("Active-inference mode requires a controller instance.")

        out.update(
            {
                "grip_open_target": float(controller.gripper_open_width),
                "grip_close_target": float(controller.gripper_close_width),
                "grip_ready_width_tol": float(controller.gripper_width_tol),
                "grip_ready_speed_tol": float(controller.gripper_speed_tol),
            }
        )
        return out

    @staticmethod
    def _fmt_vec3(vec):
        v = np.asarray(vec, dtype=float).reshape(3)
        return f"[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"

    def _ai_goal_world_xyz(self):
        goal_world = np.asarray(
            self.active_inference_params.get("place_goal_world_xyz", self.simulator.get_target_position()),
            dtype=float,
        ).reshape(3)
        pose6_raw = self.active_inference_params.get("place_goal_world_pose6d_deg", None)
        if pose6_raw is None:
            return goal_world
        try:
            pose6 = np.asarray(pose6_raw, dtype=float).reshape(-1)
            if pose6.shape[0] == 6 and np.all(np.isfinite(pose6)):
                return pose6[:3].astype(float).copy()
        except (TypeError, ValueError):
            pass
        return goal_world

    def _ai_goal_yaw_deg(self):
        yaw_deg = float(self.active_inference_params.get("place_goal_world_yaw_deg", 0.0))
        pose6_raw = self.active_inference_params.get("place_goal_world_pose6d_deg", None)
        if pose6_raw is None:
            return yaw_deg
        try:
            pose6 = np.asarray(pose6_raw, dtype=float).reshape(-1)
            if pose6.shape[0] == 6 and np.all(np.isfinite(pose6)):
                return float(pose6[5])
        except (TypeError, ValueError):
            pass
        return yaw_deg

    def runtime_debug_lines(self):
        """
        Human-readable runtime config snapshot shown at startup.
        Keeps run-time intent visible without opening multiple config files.
        """
        target_world = np.asarray(self.simulator.get_target_position(), dtype=float)
        lines = []
        if self.control_mode == "active_inference":
            p = self.active_inference_params
            goal_world = self._ai_goal_world_xyz()
            goal_yaw_deg = self._ai_goal_yaw_deg()
            lines.append(
                "[Config-AI] pick_refs_local "
                f"reach={self._fmt_vec3(p['reach_obj_rel'])} "
                f"align={self._fmt_vec3(p['align_obj_rel'])} "
                f"descend={self._fmt_vec3(p['descend_obj_rel'])}"
            )
            lines.append(
                "[Config-AI] place_refs "
                f"preplace={self._fmt_vec3(p['preplace_target_rel'])} "
                f"place={self._fmt_vec3(p['place_target_rel'])} "
                f"target_world={self._fmt_vec3(target_world)}"
            )
            lines.append(
                "[Config-AI] place_goal_pose "
                f"use_world={int(bool(p.get('use_world_place_goal_pose', False)))} "
                f"xyz={self._fmt_vec3(goal_world)} "
                f"yaw_deg={float(goal_yaw_deg):.1f} "
                f"yaw_gate={int(bool(p.get('place_goal_yaw_enabled', False)))} "
                f"yaw_thr_deg={float(p.get('place_goal_yaw_threshold_deg', 0.0)):.1f}"
            )
            lines.append(
                "[Config-AI] gates "
                f"reach={float(p['approach_threshold']):.4f} "
                f"align={float(p['align_threshold']):.4f} "
                f"desc(x/y/z)=({float(p['descend_x_threshold']):.4f}/"
                f"{float(p['descend_y_threshold']):.4f}/"
                f"{float(p['descend_z_threshold']):.4f}) "
                f"desc_norm={float(p['descend_threshold']):.4f} "
                f"place(xy/z)=({float(p['place_xy_threshold']):.4f}/"
                f"{float(p['place_z_threshold']):.4f})"
            )
            lines.append(
                "[Config-Frame] AI reach/descend errors are evaluated in world-frame "
                "object-relative coordinates; configured pick refs are object-local and "
                "converted to world each step."
            )
        else:
            t = self.task_cfg
            lines.append(
                "[Config-FSM] pick_refs "
                f"pregrasp={self._fmt_vec3(t.pregrasp_obj_rel)} "
                f"grasp={self._fmt_vec3(t.grasp_obj_rel)}"
            )
            lines.append(
                "[Config-FSM] place_refs "
                f"preplace={self._fmt_vec3(t.preplace_target_rel)} "
                f"place={self._fmt_vec3(t.place_target_rel)} "
                f"target_world={self._fmt_vec3(target_world)}"
            )
            lines.append(
                "[Config-FSM] gates "
                f"reach(xy/z)=({float(t.reach_xy_threshold):.4f}/{float(t.reach_z_threshold):.4f}) "
                f"desc(x/y/z)=({float(t.descend_x_threshold):.4f}/"
                f"{float(t.descend_y_threshold):.4f}/"
                f"{float(t.descend_z_threshold):.4f}) "
                f"desc_norm={float(t.descend_threshold):.4f} "
                f"place(xy/z)=({float(t.place_xy_threshold):.4f}/{float(t.place_z_threshold):.4f})"
            )
        return lines

    def _collect_log_metrics(self, sim_state, observation):
        reach_ref, descend_ref = self._get_references()
        obj_rel = np.asarray(observation.get("o_obj", [0.0, 0.0, 0.0]), dtype=float)
        target_rel = np.asarray(observation.get("o_target", [0.0, 0.0, 0.0]), dtype=float)
        if self.control_mode == "active_inference" and bool(
            self.active_inference_params.get("use_world_place_goal_pose", False)
        ):
            goal_world = self._ai_goal_world_xyz()
            ee_obs = np.asarray(observation.get("o_ee", [0.0, 0.0, 0.0]), dtype=float)
            target_rel = goal_world - ee_obs
        if self.control_mode == "active_inference":
            preplace_ref = np.asarray(
                self.active_inference_params.get("preplace_target_rel", self.task_cfg.preplace_target_rel),
                dtype=float,
            )
            place_ref = np.asarray(
                self.active_inference_params.get("place_target_rel", self.task_cfg.place_target_rel),
                dtype=float,
            )
        else:
            preplace_ref = self.task_cfg.preplace_target_rel
            place_ref = self.task_cfg.place_target_rel
        obs_obj_yaw = self._obs_obj_yaw(observation)
        ee_pos = np.asarray(sim_state.get("ee_pos", [0.0, 0.0, 0.0]), dtype=float)
        obj_pos = np.asarray(sim_state.get("obj_pos", [0.0, 0.0, 0.0]), dtype=float)
        if self.control_mode == "active_inference" and bool(
            self.active_inference_params.get("use_world_place_goal_pose", False)
        ):
            target_pos = self._ai_goal_world_xyz()
        else:
            target_pos = np.asarray(sim_state.get("target_pos", [0.0, 0.0, 0.0]), dtype=float)
        true_obj_yaw = self._true_obj_yaw(sim_state)
        true_obj_rel = obj_pos - ee_pos
        true_target_rel = target_pos - ee_pos

        # Reach/Descend evaluation frame:
        # - AI mode uses world-frame object-relative vectors and dynamic refs
        #   already transformed from object-local targets.
        # - FSM mode keeps existing configurable object-local/world-frame option.
        if self.control_mode == "active_inference":
            obj_rel_eval = obj_rel.copy()
            true_obj_rel_eval = true_obj_rel.copy()
        else:
            obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, obs_obj_yaw)
            true_obj_rel_eval = self._obj_rel_for_reach_descend(true_obj_rel, true_obj_yaw)

        true_reach_err_vec = true_obj_rel_eval - reach_ref
        true_descend_err_vec = true_obj_rel_eval - descend_ref
        true_preplace_err_vec = true_target_rel - preplace_ref
        true_place_err_vec = true_target_rel - place_ref
        obs_reach_err_vec = obj_rel_eval - reach_ref
        obs_descend_err_vec = obj_rel_eval - descend_ref
        obs_preplace_err_vec = target_rel - preplace_ref
        obs_place_err_vec = target_rel - place_ref

        return {
            "obs_reach_x_error": float(abs(obs_reach_err_vec[0])),
            "obs_reach_y_error": float(abs(obs_reach_err_vec[1])),
            "obs_reach_z_error": float(abs(obs_reach_err_vec[2])),
            "obs_reach_xy_error": float(np.linalg.norm(obs_reach_err_vec[:2])),
            "obs_descend_x_error": float(abs(obs_descend_err_vec[0])),
            "obs_descend_y_error": float(abs(obs_descend_err_vec[1])),
            "obs_descend_z_error": float(abs(obs_descend_err_vec[2])),
            "obs_descend_xy_error": float(np.linalg.norm(obs_descend_err_vec[:2])),
            "obs_preplace_x_error": float(abs(obs_preplace_err_vec[0])),
            "obs_preplace_y_error": float(abs(obs_preplace_err_vec[1])),
            "obs_preplace_z_error": float(abs(obs_preplace_err_vec[2])),
            "obs_preplace_xy_error": float(np.linalg.norm(obs_preplace_err_vec[:2])),
            "obs_place_x_error": float(abs(obs_place_err_vec[0])),
            "obs_place_y_error": float(abs(obs_place_err_vec[1])),
            "obs_place_z_error": float(abs(obs_place_err_vec[2])),
            "obs_place_xy_error": float(np.linalg.norm(obs_place_err_vec[:2])),
            "true_reach_x_error": float(abs(true_reach_err_vec[0])),
            "true_reach_y_error": float(abs(true_reach_err_vec[1])),
            "true_reach_z_error": float(abs(true_reach_err_vec[2])),
            "true_reach_xy_error": float(np.linalg.norm(true_reach_err_vec[:2])),
            "true_descend_x_error": float(abs(true_descend_err_vec[0])),
            "true_descend_y_error": float(abs(true_descend_err_vec[1])),
            "true_descend_z_error": float(abs(true_descend_err_vec[2])),
            "true_descend_xy_error": float(np.linalg.norm(true_descend_err_vec[:2])),
            "true_preplace_x_error": float(abs(true_preplace_err_vec[0])),
            "true_preplace_y_error": float(abs(true_preplace_err_vec[1])),
            "true_preplace_z_error": float(abs(true_preplace_err_vec[2])),
            "true_preplace_xy_error": float(np.linalg.norm(true_preplace_err_vec[:2])),
            "true_place_x_error": float(abs(true_place_err_vec[0])),
            "true_place_y_error": float(abs(true_place_err_vec[1])),
            "true_place_z_error": float(abs(true_place_err_vec[2])),
            "true_place_xy_error": float(np.linalg.norm(true_place_err_vec[:2])),
            "obs_preplace_error": float(np.linalg.norm(target_rel - preplace_ref)),
            "obs_place_error": float(np.linalg.norm(target_rel - place_ref)),
            "true_ee_z": float(ee_pos[2]),
            "true_obj_z": float(obj_pos[2]),
            "workspace_min_z": float(
                np.asarray(
                    getattr(self.simulator, "workspace_min", np.array([0.0, 0.0, 0.0], dtype=float)),
                    dtype=float,
                )[2]
            ),
        }

    def _descend_gate_for_logging(self, observation):
        if self.last_descend_gate is not None:
            return self.last_descend_gate
        if self.task_state is None or Phase(self.task_state.phase) != Phase.Descend:
            return None
        return self._compute_descend_gate(
            obj_rel=observation.get("o_obj", [0.0, 0.0, 0.0]),
            obj_yaw=self._obs_obj_yaw(observation),
            step_in_phase=int(self.task_state.step_in_phase),
            descend_contact_hold=int(self.task_state.descend_contact_hold),
            descend_ready_counter=int(self.task_state.descend_ready_counter),
        )

    def _log_heartbeat(self, sim_state, observation, action):
        phase = self._phase_name()
        contact = int(observation.get("o_contact", 0))
        grip_cmd = int(action.get("grip", 0))
        m = self._collect_log_metrics(sim_state, observation)
        obs_dt_s = self._safe_float_scalar(observation.get("o_dt", 0.0), default=0.0)
        prev_hb = self._hb_prev_metrics
        hb_step_span = 0
        d_reach_xy = 0.0
        d_desc_x = 0.0
        d_desc_y = 0.0
        d_desc_z = 0.0
        if prev_hb is not None:
            hb_step_span = max(1, int(self.step_count - prev_hb["step"]))
            d_reach_xy = float(m["true_reach_xy_error"] - prev_hb["true_reach_xy_error"])
            d_desc_x = float(m["true_descend_x_error"] - prev_hb["true_descend_x_error"])
            d_desc_y = float(m["true_descend_y_error"] - prev_hb["true_descend_y_error"])
            d_desc_z = float(m["true_descend_z_error"] - prev_hb["true_descend_z_error"])
        msg = f"[HB] step={self.step_count} phase={phase} contact={contact} grip={grip_cmd}"
        msg += (
            f" obs_age={self.obs_age_ms:.1f}ms stale={self.obs_stale_warn} "
            f"loop={self.loop_dt_ms:.1f}ms jit={self.loop_jitter_ms:.1f}ms "
            f"obs_dt={obs_dt_s:.4f}s"
        )

        if phase == "Reach" and self.control_mode == "active_inference" and self.current_belief is not None:
            turn_sign = int(self.current_belief.get("reach_turn_sign", 0))
            watchdog = int(self.current_belief.get("reach_watchdog_active", 0))
            no_prog = int(self.current_belief.get("reach_no_progress_steps", 0))
            best_err = float(self.current_belief.get("reach_best_error", float("nan")))
            yaw_align = int(self.current_belief.get("reach_yaw_align_active", 0))
            yaw_timer = int(self.current_belief.get("reach_yaw_align_timer", 0))
            approach_side_sign = int(self.current_belief.get("approach_side_sign", 0))
            reach_ref_local = np.asarray(
                self.current_belief.get(
                    "reach_obj_rel_local",
                    self.active_inference_params.get("reach_obj_rel", [0.0, 0.0, 0.0]),
                ),
                dtype=float,
            )
            reach_ref_world = np.asarray(
                self.current_belief.get(
                    "reach_obj_rel",
                    self.active_inference_params.get("reach_obj_rel", [0.0, 0.0, 0.0]),
                ),
                dtype=float,
            )
            reach_thr = float(self.active_inference_params["approach_threshold"])
            msg += (
                f" x={m['true_reach_x_error']:.4f} "
                f"y={m['true_reach_y_error']:.4f} "
                f"xy={m['true_reach_xy_error']:.4f}/{reach_thr:.4f} "
                f"z={m['true_reach_z_error']:.4f}/{reach_thr:.4f} "
                f"turn={turn_sign:+d} wd={watchdog} stall={no_prog} "
                f"best={best_err:.4f} yaw_align={yaw_align}:{yaw_timer} "
                f"ref=s{approach_side_sign:+d}"
                f"L({reach_ref_local[0]:+.3f},{reach_ref_local[1]:+.3f},{reach_ref_local[2]:+.3f})"
                f"W({reach_ref_world[0]:+.3f},{reach_ref_world[1]:+.3f},{reach_ref_world[2]:+.3f})"
            )
            if hb_step_span > 0:
                msg += f" dxy={d_reach_xy:+.4f}/{hb_step_span}"
        elif phase == Phase.ReachAbove.value:
            msg += (
                f" x={m['true_reach_x_error']:.4f} "
                f"y={m['true_reach_y_error']:.4f} "
                f"xy={m['true_reach_xy_error']:.4f}/{self.task_cfg.reach_xy_threshold:.4f} "
                f"z={m['true_reach_z_error']:.4f}/{self.task_cfg.reach_z_threshold:.4f}"
            )

        elif phase == Phase.Descend.value:
            if self.control_mode == "active_inference":
                ai_descend_x_thr = float(
                    self.active_inference_params.get("descend_x_threshold", self.task_cfg.descend_x_threshold)
                )
                ai_descend_y_thr = float(
                    self.active_inference_params.get("descend_y_threshold", self.task_cfg.descend_y_threshold)
                )
                ai_descend_z_thr = float(
                    self.active_inference_params.get("descend_z_threshold", self.task_cfg.descend_z_threshold)
                )
                ai_descend_thr = float(
                    self.active_inference_params.get("descend_threshold", self.task_cfg.descend_threshold)
                )
                ai_descend_step = int(self.current_belief.get("descend_timer", 0)) if self.current_belief else 0
                ai_descend_max_steps = int(self.active_inference_params.get("descend_max_steps", 0))
                ai_descend_no_progress = (
                    int(self.current_belief.get("descend_no_progress_steps", 0)) if self.current_belief else 0
                )
                ai_descend_ext = (
                    int(self.current_belief.get("descend_timeout_extensions", 0)) if self.current_belief else 0
                )
                ai_descend_ext_max = int(self.active_inference_params.get("descend_max_timeout_extensions", 0))
                msg += (
                    f" x={m['true_descend_x_error']:.4f}/{ai_descend_x_thr:.4f} "
                    f"y={m['true_descend_y_error']:.4f}/{ai_descend_y_thr:.4f} "
                    f"xy={m['true_descend_xy_error']:.4f}/{ai_descend_thr:.4f} "
                    f"z={m['true_descend_z_error']:.4f}/{ai_descend_z_thr:.4f} "
                    f"step={ai_descend_step}/{ai_descend_max_steps} "
                    f"stall={ai_descend_no_progress} "
                    f"ext={ai_descend_ext}/{ai_descend_ext_max}"
                )
                if hb_step_span > 0:
                    msg += (
                        f" d=({d_desc_x:+.4f},{d_desc_y:+.4f},{d_desc_z:+.4f})/"
                        f"{hb_step_span}"
                    )
            else:
                msg += (
                    f" x={m['true_descend_x_error']:.4f}/{self.task_cfg.descend_x_threshold:.4f} "
                    f"y={m['true_descend_y_error']:.4f}/{self.task_cfg.descend_y_threshold:.4f} "
                    f"xy={m['true_descend_xy_error']:.4f}/{self.task_cfg.descend_threshold:.4f} "
                    f"z={m['true_descend_z_error']:.4f}/{self.task_cfg.descend_z_threshold:.4f}"
                )
                gate = self._descend_gate_for_logging(observation)
                if gate is not None:
                    blockers = ",".join(gate.get("blockers", [])) if gate.get("blockers") else "-"
                    msg += (
                        f" ee_z={m['true_ee_z']:.4f} obj_z={m['true_obj_z']:.4f} min_z={m['workspace_min_z']:.4f} "
                        f"gate={gate['trigger']} "
                        f"ready={gate['ready_counter']}/{self.task_cfg.descend_ready_steps} "
                        f"hold={gate['contact_hold']}/{self.task_cfg.descend_stop_contact_steps} "
                        f"step={gate['step_in_phase']}/{self.task_cfg.descend_max_steps} "
                        f"blockers={blockers}"
                    )
        elif phase == Phase.Close.value and self.task_state is not None:
            msg += (
                f" x={m['true_descend_x_error']:.4f}/{self.task_cfg.descend_x_threshold:.4f} "
                f"y={m['true_descend_y_error']:.4f}/{self.task_cfg.descend_y_threshold:.4f} "
                f"z={m['true_descend_z_error']:.4f}/{self.task_cfg.descend_z_threshold:.4f} "
                f" close_step={int(self.task_state.step_in_phase)}/{self.task_cfg.close_hold_steps} "
                f"stable_contact={int(self.task_state.stable_contact_counter)}/{self.task_cfg.stable_contact_steps}"
            )
        elif phase == Phase.LiftTest.value and self.task_state is not None:
            msg += (
                f" lift_step={int(self.task_state.lift_test_timer)}/{self.task_cfg.lift_test_steps} "
                f"ee_z={m['true_ee_z']:.4f} obj_z={m['true_obj_z']:.4f}"
            )
        elif phase == Phase.Transit.value:
            transit_target = (
                float(self.active_inference_params.get("transit_height", self.task_cfg.transit_height))
                if self.control_mode == "active_inference"
                else self.task_cfg.transit_height
            )
            msg += f" ee_z={m['true_ee_z']:.4f} transit_target_z={transit_target:.4f}"
        elif phase == Phase.MoveToPlaceAbove.value:
            pre_xy_thr = (
                float(self.active_inference_params.get("preplace_xy_threshold", self.task_cfg.preplace_xy_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.preplace_xy_threshold
            )
            pre_z_thr = (
                float(self.active_inference_params.get("preplace_z_threshold", self.task_cfg.preplace_z_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.preplace_z_threshold
            )
            pre_thr = (
                float(self.active_inference_params.get("preplace_threshold", self.task_cfg.preplace_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.preplace_threshold
            )
            msg += (
                f" x={m['true_preplace_x_error']:.4f} "
                f"y={m['true_preplace_y_error']:.4f} "
                f"xy={m['true_preplace_xy_error']:.4f}/{pre_xy_thr:.4f} "
                f"z={m['true_preplace_z_error']:.4f}/{pre_z_thr:.4f} "
                f"norm={m['obs_preplace_error']:.4f}/{pre_thr:.4f} "
                f"ee_z={m['true_ee_z']:.4f}"
            )
        elif phase == Phase.DescendToPlace.value:
            place_xy_thr = (
                float(self.active_inference_params.get("place_xy_threshold", self.task_cfg.place_xy_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.place_xy_threshold
            )
            place_z_thr = (
                float(self.active_inference_params.get("place_z_threshold", self.task_cfg.place_z_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.place_z_threshold
            )
            place_thr = (
                float(self.active_inference_params.get("place_threshold", self.task_cfg.place_threshold))
                if self.control_mode == "active_inference"
                else self.task_cfg.place_threshold
            )
            msg += (
                f" x={m['true_place_x_error']:.4f} "
                f"y={m['true_place_y_error']:.4f} "
                f"xy={m['true_place_xy_error']:.4f}/{place_xy_thr:.4f} "
                f"z={m['true_place_z_error']:.4f}/{place_z_thr:.4f} "
                f"norm={m['obs_place_error']:.4f}/{place_thr:.4f} "
                f"ee_z={m['true_ee_z']:.4f} obj_z={m['true_obj_z']:.4f}"
            )
        elif self.control_mode == "active_inference" and phase == "Open" and self.current_belief is not None:
            open_timer = int(self.current_belief.get("open_timer", 0))
            open_steps = int(self.active_inference_params.get("open_hold_steps", 0))
            msg += (
                f" open_step={open_timer}/{open_steps} "
                f"x={m['true_place_x_error']:.4f} y={m['true_place_y_error']:.4f} z={m['true_place_z_error']:.4f}"
            )
        elif self.control_mode == "active_inference" and phase == "Retreat" and self.current_belief is not None:
            retreat_timer = int(self.current_belief.get("retreat_timer", 0))
            retreat_steps = int(self.active_inference_params.get("retreat_steps", 0))
            msg += f" retreat_step={retreat_timer}/{retreat_steps}"
        elif self.control_mode == "active_inference" and phase == "Done":
            msg += " status=done"
        elif phase == Phase.Open.value and self.task_state is not None:
            msg += (
                f" open_step={int(self.task_state.step_in_phase)}/{self.task_cfg.open_hold_steps} "
                f"x={m['true_place_x_error']:.4f} y={m['true_place_y_error']:.4f} z={m['true_place_z_error']:.4f}"
            )
        elif phase == Phase.Retreat.value and self.task_state is not None:
            msg += f" retreat_step={int(self.task_state.step_in_phase)}/{self.task_cfg.retreat_steps}"
        else:
            msg += (
                f" reach=({m['true_reach_x_error']:.4f},{m['true_reach_y_error']:.4f},{m['true_reach_z_error']:.4f}) "
                f"desc=({m['true_descend_x_error']:.4f},{m['true_descend_y_error']:.4f},{m['true_descend_z_error']:.4f})"
            )

        if self.control_mode == "active_inference" and self.current_belief is not None:
            conf = float(self.current_belief.get("obs_confidence", 1.0))
            vfe = float(self.current_belief.get("vfe_total", 0.0))
            conf_ok = int(self.current_belief.get("phase_conf_ok", 1))
            vfe_ok = int(self.current_belief.get("phase_vfe_ok", 1))
            msg += f" conf={conf:.2f}"
            msg += f" vfe={vfe:.3f}"
            msg += f" gate={conf_ok}:{vfe_ok}"
            msg += f" overrun={self.loop_overrun_count}"
            release_warn = int(self.current_belief.get("release_warning", 0))
            if phase == "Open" or release_warn == 1:
                release_counter = int(self.current_belief.get("release_contact_counter", 0))
                release_warn_steps = int(self.active_inference_params.get("release_contact_warn_steps", 0))
                detach_counter = int(self.current_belief.get("release_detach_counter", 0))
                stable_counter = int(self.current_belief.get("release_stable_counter", 0))
                detach_steps = int(self.active_inference_params.get("release_detach_hold_steps", 0))
                stable_steps = int(self.active_inference_params.get("release_stable_hold_steps", 0))
                msg += (
                    f" release_hold={release_counter}/{release_warn_steps} "
                    f"detach={detach_counter}/{detach_steps} "
                    f"stable={stable_counter}/{stable_steps} "
                    f"release_warn={release_warn}"
                )
            if self.ai_risk_detection_enabled:
                rs = self.ai_risk_state
                msg += (
                    f" dq_ratio={float(rs.get('dq_ratio', 0.0)):.2f} "
                    f"sing={int(rs.get('singularity_warn', 0))} "
                    f"uc={int(rs.get('unintended_contact_warn', 0))}"
                )

        if self.control_mode == "active_inference" and self.ai_bt is not None:
            bt_reason = self.ai_bt.last_reason if self.ai_bt.last_reason else "-"
            msg += f" bt={self.ai_bt.status.value} bt_reason={bt_reason}"
            if self.current_belief is not None:
                retry_reason = str(self.current_belief.get("last_retry_reason", "")).strip()
                fail_reason = str(self.current_belief.get("failure_reason", "")).strip()
                recovery_branch = str(self.current_belief.get("recovery_branch", "")).strip()
                recovery_global = int(self.current_belief.get("recovery_global_count", 0))
                recovery_branch_retry = int(self.current_belief.get("recovery_branch_retry", 0))
                if recovery_branch:
                    msg += (
                        f" branch={recovery_branch}"
                        f" br_retry={recovery_branch_retry}"
                        f" recov={recovery_global}"
                    )
                if retry_reason:
                    msg += f" retry_reason={retry_reason}"
                if fail_reason:
                    msg += f" fail_reason={fail_reason}"

        if self.log_pose_debug:
            hand_roll, hand_pitch, hand_yaw = self._quat_wxyz_to_rpy(
                sim_state.get("hand_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
            )
            joint7_pos = float(sim_state.get("joint7_pos", 0.0))
            msg += (
                f" joint7={joint7_pos:.4f} "
                f"rpy=({hand_roll:.4f},{hand_pitch:.4f},{hand_yaw:.4f})"
            )

        self._hb_prev_metrics = {
            "step": int(self.step_count),
            "true_reach_xy_error": float(m["true_reach_xy_error"]),
            "true_descend_x_error": float(m["true_descend_x_error"]),
            "true_descend_y_error": float(m["true_descend_y_error"]),
            "true_descend_z_error": float(m["true_descend_z_error"]),
        }
        print(msg)

    def _log_step(self, sim_state, observation, action):
        ee = np.asarray(observation["o_ee"], dtype=float)
        obj_rel = np.asarray(observation["o_obj"], dtype=float)
        if self.control_mode == "active_inference":
            # AI pick refs are transformed to world-frame relation each step.
            # Keep logged reach/descend errors in the same frame as controller math.
            obj_rel_eval = obj_rel.copy()
        else:
            obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, self._obs_obj_yaw(observation))
        target_rel = np.asarray(observation["o_target"], dtype=float)
        pregrasp_ref, descend_ref = self._get_references()
        pregrasp_error = float(np.linalg.norm(obj_rel_eval - pregrasp_ref))
        descend_error = float(np.linalg.norm(obj_rel_eval - descend_ref))
        descend_x_error = float(abs((obj_rel_eval - descend_ref)[0]))
        descend_y_error = float(abs((obj_rel_eval - descend_ref)[1]))
        descend_z_error = float(abs((obj_rel_eval - descend_ref)[2]))
        if "ee_target_pos" in action:
            move = np.asarray(action["ee_target_pos"], dtype=float) - np.asarray(sim_state["ee_pos"], dtype=float)
        else:
            move = np.asarray(action.get("move", [0.0, 0.0, 0.0]), dtype=float)

        obs_dt_s = self._safe_float_scalar(observation.get("o_dt", 0.0), default=0.0)
        ee_vel_obs = np.asarray(observation.get("o_ee_vel", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        obj_vel_obs = np.asarray(observation.get("o_obj_vel", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        tgt_vel_obs = np.asarray(observation.get("o_target_vel", [0.0, 0.0, 0.0]), dtype=float).reshape(-1)
        ee_vel_norm = self._safe_norm(ee_vel_obs[:3]) if ee_vel_obs.size >= 3 else 0.0
        obj_vel_norm = self._safe_norm(obj_vel_obs[:3]) if obj_vel_obs.size >= 3 else 0.0
        tgt_vel_norm = self._safe_norm(tgt_vel_obs[:3]) if tgt_vel_obs.size >= 3 else 0.0
        m = self._collect_log_metrics(sim_state, observation)

        if self.control_mode == "active_inference" and self.current_belief is not None:
            phase_name = str(self.current_belief.get("phase", "Reach"))
            phase_timer_key = {
                "Align": "align_timer",
                "PreGraspHold": "pregrasp_hold_timer",
                "Descend": "descend_timer",
                "CloseHold": "close_hold_timer",
                "LiftTest": "lift_test_timer",
                "Transit": "transit_timer",
                "DescendToPlace": "place_descend_timer",
                "Open": "open_timer",
                "Retreat": "retreat_timer",
            }.get(phase_name, "")
            phase_step = int(self.current_belief.get(phase_timer_key, 0)) if phase_timer_key else 0
        else:
            phase_name = self.task_state.phase.value if self.task_state is not None else Phase.ReachAbove.value
            phase_step = int(self.task_state.step_in_phase) if self.task_state is not None else 0

        row = {
            "step": self.step_count,
            "phase": phase_name,
            "sim_time": self._safe_float_scalar(sim_state.get("sim_time", 0.0), default=0.0),
            "obs_timestamp": float(self.obs_timestamp),
            "obs_age_ms": float(self.obs_age_ms),
            "obs_stale_warn": int(self.obs_stale_warn),
            "loop_dt_ms": float(self.loop_dt_ms),
            "loop_dt_ema_ms": float(self.loop_dt_ema_ms),
            "loop_jitter_ms": float(self.loop_jitter_ms),
            "loop_target_ms": float(self.loop_target_ms),
            "loop_overrun_count": int(self.loop_overrun_count),
            "obs_dt_s": float(obs_dt_s),
            "est_ee_vel_norm": float(ee_vel_norm),
            "est_obj_vel_norm": float(obj_vel_norm),
            "est_target_vel_norm": float(tgt_vel_norm),
            "distance_to_object": float(np.linalg.norm(obj_rel)),
            "reach_error": pregrasp_error,
            "pregrasp_error": pregrasp_error,
            "descend_error": descend_error,
            "descend_x_error": descend_x_error,
            "descend_y_error": descend_y_error,
            "descend_z_error": descend_z_error,
            "true_reach_xy_error": float(m["true_reach_xy_error"]),
            "true_descend_xy_error": float(m["true_descend_xy_error"]),
            "obs_reach_xy_error": float(m["obs_reach_xy_error"]),
            "obs_descend_xy_error": float(m["obs_descend_xy_error"]),
            "obs_preplace_error": float(m["obs_preplace_error"]),
            "obs_place_error": float(m["obs_place_error"]),
            "active_reach_ref_x": float(pregrasp_ref[0]),
            "active_reach_ref_y": float(pregrasp_ref[1]),
            "active_reach_ref_z": float(pregrasp_ref[2]),
            "active_descend_ref_x": float(descend_ref[0]),
            "active_descend_ref_y": float(descend_ref[1]),
            "active_descend_ref_z": float(descend_ref[2]),
            "phase_step": int(phase_step),
            "distance_to_target": float(np.linalg.norm(target_rel)),
            "s_ee_x": float(ee[0]),
            "s_ee_y": float(ee[1]),
            "s_ee_z": float(ee[2]),
            "s_obj_rel_x": float(obj_rel[0]),
            "s_obj_rel_y": float(obj_rel[1]),
            "s_obj_rel_z": float(obj_rel[2]),
            "s_target_rel_x": float(target_rel[0]),
            "s_target_rel_y": float(target_rel[1]),
            "s_target_rel_z": float(target_rel[2]),
            "action_move_x": float(move[0]),
            "action_move_y": float(move[1]),
            "action_move_z": float(move[2]),
            "action_grip": int(action["grip"]),
            "escape_active": int(self.escape_active),
            "obs_contact": int(observation["o_contact"]),
            "obs_grip": float(observation["o_grip"]),
            "true_ee_x": float(sim_state["ee_pos"][0]),
            "true_ee_y": float(sim_state["ee_pos"][1]),
            "true_ee_z": float(sim_state["ee_pos"][2]),
            "true_obj_x": float(sim_state["obj_pos"][0]),
            "true_obj_y": float(sim_state["obj_pos"][1]),
            "true_obj_z": float(sim_state["obj_pos"][2]),
            "true_target_x": float(sim_state["target_pos"][0]),
            "true_target_y": float(sim_state["target_pos"][1]),
            "true_target_z": float(sim_state["target_pos"][2]),
            "true_joint7_pos": float(sim_state.get("joint7_pos", 0.0)),
        }
        if self.control_mode == "active_inference" and self.current_belief is not None:
            b_ee = np.asarray(self.current_belief.get("s_ee_mean", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            b_obj = np.asarray(self.current_belief.get("s_obj_mean", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            b_tgt = np.asarray(self.current_belief.get("s_target_mean", [np.nan, np.nan, np.nan]), dtype=float).reshape(-1)
            b_ee = b_ee if b_ee.size >= 3 else np.array([np.nan, np.nan, np.nan], dtype=float)
            b_obj = b_obj if b_obj.size >= 3 else np.array([np.nan, np.nan, np.nan], dtype=float)
            b_tgt = b_tgt if b_tgt.size >= 3 else np.array([np.nan, np.nan, np.nan], dtype=float)
            row["ai_belief_ee_x"] = float(b_ee[0])
            row["ai_belief_ee_y"] = float(b_ee[1])
            row["ai_belief_ee_z"] = float(b_ee[2])
            row["ai_belief_obj_rel_x"] = float(b_obj[0])
            row["ai_belief_obj_rel_y"] = float(b_obj[1])
            row["ai_belief_obj_rel_z"] = float(b_obj[2])
            row["ai_belief_target_rel_x"] = float(b_tgt[0])
            row["ai_belief_target_rel_y"] = float(b_tgt[1])
            row["ai_belief_target_rel_z"] = float(b_tgt[2])
            row["ai_belief_obj_yaw"] = float(self.current_belief.get("s_obj_yaw", np.nan))
            row["ai_obs_confidence"] = float(self.current_belief.get("obs_confidence", 1.0))
            row["ai_vfe_total"] = float(self.current_belief.get("vfe_total", 0.0))
            row["ai_phase_conf_ok"] = int(self.current_belief.get("phase_conf_ok", 1))
            row["ai_phase_vfe_ok"] = int(self.current_belief.get("phase_vfe_ok", 1))
            row["ai_phase_gate_ok"] = int(self.current_belief.get("phase_gate_ok", 1))
            row["ai_bt_status"] = (
                self.ai_bt.status.value if (self.ai_bt is not None and self.ai_bt.status is not None) else ""
            )
            row["ai_bt_reason"] = self.ai_bt.last_reason if self.ai_bt is not None else ""
            row["ai_retry_reason"] = str(self.current_belief.get("last_retry_reason", ""))
            row["ai_failure_reason"] = str(self.current_belief.get("failure_reason", ""))
            row["ai_recovery_branch"] = str(self.current_belief.get("recovery_branch", ""))
            row["ai_recovery_branch_retry"] = int(self.current_belief.get("recovery_branch_retry", 0))
            row["ai_recovery_global_count"] = int(self.current_belief.get("recovery_global_count", 0))
            row["ai_release_detach_counter"] = int(self.current_belief.get("release_detach_counter", 0))
            row["ai_release_stable_counter"] = int(self.current_belief.get("release_stable_counter", 0))
            row["ai_release_reapproach_count"] = int(
                self.current_belief.get("release_reapproach_count", 0)
            )
        else:
            row["ai_belief_ee_x"] = ""
            row["ai_belief_ee_y"] = ""
            row["ai_belief_ee_z"] = ""
            row["ai_belief_obj_rel_x"] = ""
            row["ai_belief_obj_rel_y"] = ""
            row["ai_belief_obj_rel_z"] = ""
            row["ai_belief_target_rel_x"] = ""
            row["ai_belief_target_rel_y"] = ""
            row["ai_belief_target_rel_z"] = ""
            row["ai_belief_obj_yaw"] = ""
            row["ai_obs_confidence"] = ""
            row["ai_vfe_total"] = ""
            row["ai_phase_conf_ok"] = ""
            row["ai_phase_vfe_ok"] = ""
            row["ai_phase_gate_ok"] = ""
            row["ai_bt_status"] = ""
            row["ai_bt_reason"] = ""
            row["ai_retry_reason"] = ""
            row["ai_failure_reason"] = ""
            row["ai_recovery_branch"] = ""
            row["ai_recovery_branch_retry"] = ""
            row["ai_recovery_global_count"] = ""
            row["ai_release_detach_counter"] = ""
            row["ai_release_stable_counter"] = ""
            row["ai_release_reapproach_count"] = ""
        hand_quat = sim_state.get("hand_quat_wxyz")
        if hand_quat is not None:
            hr, hp, hy = self._quat_wxyz_to_rpy(hand_quat)
            row["true_hand_roll"] = float(hr)
            row["true_hand_pitch"] = float(hp)
            row["true_hand_yaw"] = float(hy)
        else:
            row["true_hand_roll"] = 0.0
            row["true_hand_pitch"] = 0.0
            row["true_hand_yaw"] = 0.0
        self._csv_writer.writerow(row)
        if self.step_count % 50 == 0:
            self._csv_file.flush()

    def _build_recovery_action(self, sim_state):
        """
        Short true-state guided action used only when stall is detected.
        This is a local recovery behavior; normal AI policy resumes afterward.
        """
        ee_pos = np.array(sim_state["ee_pos"], dtype=float)
        obj_pos = np.array(sim_state["obj_pos"], dtype=float)
        desired = obj_pos + np.array([0.0, 0.0, self.recovery_height], dtype=float)
        vec = desired - ee_pos
        n = float(np.linalg.norm(vec))
        if n > self.recovery_max_step and n > 0:
            vec = (vec / n) * self.recovery_max_step
        return {"move": vec.tolist(), "grip": 0}

    def _maybe_escape_stall(self, action, sim_state):
        if self.control_mode == "active_inference" and not self.ai_allow_lowlevel_escape:
            self.escape_active = 0
            self.recovery_steps_remaining = 0
            return action

        self.escape_active = 0
        in_reach = False
        if self.control_mode == "active_inference":
            in_reach = self.current_belief is not None and str(self.current_belief.get("phase")) == "Reach"
        else:
            in_reach = self.task_state is not None and Phase(self.task_state.phase) == Phase.ReachAbove

        if not in_reach:
            self.escape_cooldown = max(0, self.escape_cooldown - 1)
            self.recovery_steps_remaining = 0
            return action

        # Avoid premature recovery during initial transients.
        if self.step_count < 300:
            return action

        if self.recovery_steps_remaining > 0:
            self.recovery_steps_remaining -= 1
            self.escape_active = 2
            return self._build_recovery_action(sim_state)

        if self.escape_cooldown > 0:
            self.escape_cooldown -= 1
            return action

        if len(self.ee_true_history) < self.stall_window + 1:
            return action

        ee_start = self.ee_true_history[-(self.stall_window + 1)]
        ee_end = self.ee_true_history[-1]
        displacement = float(np.linalg.norm(ee_end - ee_start))
        # Use current observation (object relative) to decide if we are making progress.
        obj_rel = np.asarray(self.obs_history[-1].get("o_obj", [0.0, 0.0, 0.0]), dtype=float)
        obj_yaw = self._obs_obj_yaw(self.obs_history[-1])
        obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, obj_yaw)
        reach_ref, _ = self._get_references()
        reach_error = float(np.linalg.norm(obj_rel_eval - reach_ref))

        if displacement >= self.stall_disp_threshold or reach_error < 0.10:
            return action

        # Escape trigger: start a short guided recovery sequence.
        xy = obj_rel[:2]
        xy_norm = float(np.linalg.norm(xy))
        if xy_norm > 1e-6:
            xy_away = -xy / xy_norm
        else:
            xy_away = np.array([-1.0, 0.0], dtype=float)

        escape_move = np.array([0.004 * xy_away[0], 0.004 * xy_away[1], 0.009], dtype=float)
        self.escape_cooldown = self.escape_steps
        self.recovery_steps_remaining = self.recovery_steps
        self.escape_active = 1
        print(f"[Escape] stall detected at step {self.step_count}: displacement={displacement:.5f}, reach_error={reach_error:.5f}")
        return {"move": escape_move.tolist(), "grip": 0}

    def _compute_descend_gate(self, obj_rel, obj_yaw, step_in_phase, descend_contact_hold, descend_ready_counter):
        """
        Compute descend trigger/gate status using the same conditions as step_fsm.
        This is used only for explainability logs.
        """
        obj_rel = np.asarray(obj_rel, dtype=float)
        obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, float(obj_yaw))
        err = obj_rel_eval - self.task_cfg.grasp_obj_rel
        x_err = float(abs(err[0]))
        y_err = float(abs(err[1]))
        z_err = float(abs(err[2]))

        x_ok = x_err <= self.task_cfg.descend_x_threshold
        y_ok = y_err <= self.task_cfg.descend_y_threshold
        xy_ok = x_ok and y_ok
        z_ok = z_err <= self.task_cfg.descend_z_threshold
        ready_ok = int(descend_ready_counter) >= int(self.task_cfg.descend_ready_steps)

        position_ok = ((float(np.linalg.norm(err)) <= self.task_cfg.descend_threshold) or (xy_ok and z_ok)) and ready_ok
        contact_stop = (
            int(descend_contact_hold) >= int(self.task_cfg.descend_stop_contact_steps)
            and z_err <= self.task_cfg.descend_contact_z_threshold
            and ready_ok
        )
        timeout_near = (
            x_err <= self.task_cfg.descend_timeout_x_threshold
            and y_err <= self.task_cfg.descend_timeout_y_threshold
            and z_err <= self.task_cfg.descend_timeout_z_threshold
        )
        timeout_stop = int(step_in_phase) >= int(self.task_cfg.descend_max_steps) and timeout_near and ready_ok

        trigger = "none"
        if position_ok:
            trigger = "position_ok"
        elif contact_stop:
            trigger = "contact_stop"
        elif timeout_stop:
            trigger = "timeout_stop"

        blockers = []
        if trigger == "none":
            if not x_ok:
                blockers.append(f"x>{self.task_cfg.descend_x_threshold:.3f}")
            if not y_ok:
                blockers.append(f"y>{self.task_cfg.descend_y_threshold:.3f}")
            if not z_ok:
                blockers.append(f"z>{self.task_cfg.descend_z_threshold:.3f}")
            if not ready_ok:
                blockers.append(
                    f"ready={int(descend_ready_counter)}/{int(self.task_cfg.descend_ready_steps)}"
                )
            if int(step_in_phase) >= int(self.task_cfg.descend_max_steps) and not timeout_near:
                blockers.append("timeout_near=false")

        return {
            "step_in_phase": int(step_in_phase),
            "x_err": x_err,
            "y_err": y_err,
            "z_err": z_err,
            "x_ok": bool(x_ok),
            "y_ok": bool(y_ok),
            "z_ok": bool(z_ok),
            "ready_counter": int(descend_ready_counter),
            "ready_ok": bool(ready_ok),
            "contact_hold": int(descend_contact_hold),
            "trigger": trigger,
            "blockers": blockers,
        }

    def _compute_descend_gate_from_prev(self, prev_state, observation):
        """
        Evaluate descend gate for the *current* step from previous state + current observation,
        matching the exact update order used by step_fsm.
        """
        obj_rel = np.asarray(observation.get("o_obj", [0.0, 0.0, 0.0]), dtype=float)
        obj_yaw = self._obs_obj_yaw(observation)
        obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, obj_yaw)
        contact = int(observation.get("o_contact", 0))
        err = obj_rel_eval - self.task_cfg.grasp_obj_rel
        xy_ok = (
            abs(float(err[0])) <= self.task_cfg.descend_x_threshold
            and abs(float(err[1])) <= self.task_cfg.descend_y_threshold
        )
        z_ok = abs(float(err[2])) <= self.task_cfg.descend_z_threshold
        descend_contact_hold = (int(prev_state.descend_contact_hold) + 1) if contact == 1 else 0
        descend_ready_counter = (int(prev_state.descend_ready_counter) + 1) if (xy_ok and z_ok) else 0
        return self._compute_descend_gate(
            obj_rel=obj_rel,
            obj_yaw=obj_yaw,
            step_in_phase=int(prev_state.step_in_phase) + 1,
            descend_contact_hold=descend_contact_hold,
            descend_ready_counter=descend_ready_counter,
        )

    def step(self):
        self._update_runtime_timing()
        sim_state = self.simulator.get_state()
        self.ee_true_history.append(np.array(sim_state["ee_pos"], dtype=float))
        observation = self.sensor_backend.get_observation(sim_state)
        self._update_observation_freshness(sim_state, observation)
        self.obs_history.append(observation)

        if self.control_mode == "active_inference":
            if self.step_count < self.ai_startup_settle_steps:
                if not self._ai_settle_logged:
                    print(f"[AI] startup_settle_steps={self.ai_startup_settle_steps} (holding EE while object settles)")
                    self._ai_settle_logged = True
                action = {
                    "move": [0.0, 0.0, 0.0],
                    "grip": -1,
                    "enable_yaw_objective": False,
                    "enable_topdown_objective": True,
                }
                self._reset_ai_risk_state()
            else:
                self.current_belief = infer_beliefs(
                    observation=observation,
                    previous_belief=self.current_belief,
                    params=self.active_inference_params,
                )
                self._update_ai_risk_detection(observation)
                # Expose monitor signals to BT so recovery can use standard reason taxonomy.
                if self.current_belief is not None:
                    self.current_belief["obs_stale_warn"] = int(self.obs_stale_warn)
                    self.current_belief["risk_singularity_warn"] = int(
                        self.ai_risk_state.get("singularity_warn", 0)
                    )
                    self.current_belief["risk_unexpected_contact_warn"] = int(
                        self.ai_risk_state.get("unintended_contact_warn", 0)
                    )
                if self.ai_bt is not None:
                    bt_result = self.ai_bt.tick(self.current_belief)
                    if bt_result.get("recover", False):
                        self.current_belief = self.ai_bt.recover_belief(self.current_belief)
                        branch = str(self.current_belief.get("recovery_branch", ""))
                        branch_retry = int(self.current_belief.get("recovery_branch_retry", 0))
                        global_count = int(self.current_belief.get("recovery_global_count", 0))
                        print(
                            f"[AI-BT] recover reason={self.ai_bt.last_reason} "
                            f"branch={branch or '-'} "
                            f"branch_retry={branch_retry} "
                            f"global={global_count} "
                            f"retry={int(self.current_belief.get('retry_count', 0))}"
                        )
                action = select_action(self.current_belief, params=self.active_inference_params)
        else:
            prev_task_state = self.task_state
            descend_gate = None
            if prev_task_state is not None and Phase(prev_task_state.phase) == Phase.Descend:
                descend_gate = self._compute_descend_gate_from_prev(prev_task_state, observation)

            self.task_state = step_fsm(self.task_state, observation, cfg=self.task_cfg)
            if descend_gate is not None:
                self.last_descend_gate = descend_gate
                if Phase(self.task_state.phase) == Phase.Close:
                    self.last_transition_reason = str(descend_gate.get("trigger", "unknown"))
                else:
                    self.last_transition_reason = ""
            elif self.task_state is not None and Phase(self.task_state.phase) != Phase.Descend:
                self.last_descend_gate = None
                self.last_transition_reason = ""
            action = self.policy.act(self.task_state, observation)
            # Fallback trigger in case transition-event based pause misses for any reason.
            if (
                self.pause_on_reach_to_descend
                and self.task_state is not None
                and Phase(self.task_state.phase) == Phase.Descend
                and int(self.task_state.step_in_phase) == 0
                and not self.pause_pending
            ):
                self._queue_pause("ReachAbove->Descend(step0)", is_reach_to_descend=True)
        action = self._maybe_escape_stall(action, sim_state)
        self.action_history.append(action)

        self._log_step(sim_state, observation, action)
        self._log_events(sim_state, observation, action)
        self._maybe_pause_for_inspection(sim_state, observation)

        if self.step_count % self.log_every_steps == 0:
            self._log_heartbeat(sim_state, observation, action)

        self.actuator_backend.apply_action(action)
        self.step_count += 1

    def close(self):
        if hasattr(self, "_csv_file") and self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

    def _maybe_pause_for_inspection(self, sim_state, observation):
        if not self.pause_pending:
            return

        ee_pos = np.asarray(sim_state.get("ee_pos", [0.0, 0.0, 0.0]), dtype=float)
        obj_pos = np.asarray(sim_state.get("obj_pos", [0.0, 0.0, 0.0]), dtype=float)
        obj_rel = obj_pos - ee_pos
        obj_yaw = self._true_obj_yaw(sim_state)
        if self.control_mode == "active_inference":
            obj_rel_eval = obj_rel.copy()
        else:
            obj_rel_eval = self._obj_rel_for_reach_descend(obj_rel, obj_yaw)
        _, descend_ref = self._get_references()
        descend_x_error = float(abs((obj_rel_eval - descend_ref)[0]))
        descend_y_error = float(abs((obj_rel_eval - descend_ref)[1]))
        descend_z_error = float(abs((obj_rel_eval - descend_ref)[2]))
        phase = (
            self.current_belief.get("phase", "Reach")
            if self.control_mode == "active_inference" and self.current_belief is not None
            else (self.task_state.phase.value if self.task_state is not None else Phase.ReachAbove.value)
        )
        ee_z = float(ee_pos[2])
        obj_z = float(obj_pos[2])
        obj_rel_z = float(obj_rel[2])
        target_rel_z = float(descend_ref[2])

        print(
            f"[Pause] step={self.step_count} {self.pause_reason} "
            f"phase={phase} "
            f"ee_z={ee_z:.4f} obj_z={obj_z:.4f} "
            f"obj_rel_z={obj_rel_z:.4f} target_rel_z={target_rel_z:.4f} "
            f"descend_x_error={descend_x_error:.4f} "
            f"descend_y_error={descend_y_error:.4f} "
            f"descend_z_error={descend_z_error:.4f}"
        )
        try:
            input("Paused for inspection. Press Enter to continue...")
        except EOFError:
            # Non-interactive terminal: continue without blocking.
            print("[Pause] input unavailable (EOF); continuing.")
        self.pause_pending = False

    def _queue_pause(self, reason, is_reach_to_descend=False):
        if not reason:
            return
        if (
            is_reach_to_descend
            and self.pause_on_reach_to_descend_once
            and self.reach_to_descend_pause_already_triggered
        ):
            return
        if not self.pause_pending:
            self.pause_pending = True
            self.pause_reason = str(reason)
        else:
            reasons = [r.strip() for r in str(self.pause_reason).split("|") if r.strip()]
            if str(reason) not in reasons:
                reasons.append(str(reason))
                self.pause_reason = " | ".join(reasons)
        if is_reach_to_descend:
            self.reach_to_descend_pause_already_triggered = True

    def _log_events(self, sim_state, observation, action):
        """
        Event logs for key state/action transitions so run behavior is easier to inspect.
        """
        phase = self._phase_name()
        contact = int(observation.get("o_contact", 0))
        m = self._collect_log_metrics(sim_state, observation)

        if self.prev_phase is None:
            self.prev_phase = phase
            self.prev_contact = contact
            self.prev_escape_active = self.escape_active
            self.prev_action_grip = int(action.get("grip", 0))
            if self.control_mode == "active_inference" and self.current_belief is not None:
                self.prev_ai_release_warning = int(self.current_belief.get("release_warning", 0))
            else:
                self.prev_ai_release_warning = 0
            self.prev_ai_singularity_warn = int(self.ai_risk_state.get("singularity_warn", 0))
            self.prev_ai_unintended_contact_warn = int(
                self.ai_risk_state.get("unintended_contact_warn", 0)
            )
            print(
                f"[Init] phase={phase} contact={contact} "
                f"log_format=v2 "
                f"pause_on_reach_to_descend={int(self.pause_on_reach_to_descend)} "
                f"pause_once={int(self.pause_on_reach_to_descend_once)} "
                f"pause_on_phase_change={int(self.pause_on_phase_change)} "
                f"pause_on_grip_start={int(self.pause_on_grip_start)}"
            )
            return

        if phase != self.prev_phase:
            msg = (
                f"[Phase] step={self.step_count} {self.prev_phase}->{phase} contact={contact} "
                f"reach=({m['true_reach_x_error']:.4f},{m['true_reach_y_error']:.4f},{m['true_reach_z_error']:.4f}) "
                f"desc=({m['true_descend_x_error']:.4f},{m['true_descend_y_error']:.4f},{m['true_descend_z_error']:.4f})"
            )
            if self.prev_phase == Phase.Descend.value and phase == Phase.Close.value:
                gate = self.last_descend_gate
                msg += f" trigger={self.last_transition_reason or 'unknown'}"
                if gate is not None:
                    blockers = ",".join(gate.get("blockers", [])) if gate.get("blockers") else "-"
                    msg += (
                        f" gate_ready={gate['ready_counter']}/{self.task_cfg.descend_ready_steps} "
                        f"gate_hold={gate['contact_hold']}/{self.task_cfg.descend_stop_contact_steps} "
                        f"gate_blockers={blockers}"
                    )
            elif phase == Phase.Descend.value:
                msg += f" ee_z={m['true_ee_z']:.4f} obj_z={m['true_obj_z']:.4f}"
            elif phase == Phase.MoveToPlaceAbove.value:
                msg += f" preplace_err={m['obs_preplace_error']:.4f}"
            elif phase == Phase.DescendToPlace.value:
                msg += f" place_err={m['obs_place_error']:.4f}"
            elif self.control_mode == "active_inference" and phase == "Failure" and self.current_belief is not None:
                msg += f" reason={str(self.current_belief.get('failure_reason', 'phase_failure'))}"
            print(msg)
            if self.pause_on_phase_change:
                self._queue_pause(f"{self.prev_phase}->{phase}")
            if (
                self.pause_on_reach_to_descend
                and self.control_mode != "active_inference"
                and self.prev_phase == Phase.ReachAbove.value
                and phase == Phase.Descend.value
            ):
                self._queue_pause(f"{self.prev_phase}->{phase}", is_reach_to_descend=True)

        grip_cmd = int(action.get("grip", 0))
        if (
            self.pause_on_grip_start
            and self.prev_action_grip is not None
            and self.prev_action_grip != 1
            and grip_cmd == 1
        ):
            self._queue_pause(f"GripStart(phase={phase})")
            print(
                f"[GripStart] step={self.step_count} phase={phase} "
                f"desc=({m['true_descend_x_error']:.4f},{m['true_descend_y_error']:.4f},{m['true_descend_z_error']:.4f})"
            )

        if contact != self.prev_contact and self.log_contact_events:
            state = "ON" if contact == 1 else "OFF"
            print(
                f"[Contact{state}] step={self.step_count} "
                f"reach=({m['obs_reach_x_error']:.4f},{m['obs_reach_y_error']:.4f},{m['obs_reach_z_error']:.4f}) "
                f"desc=({m['obs_descend_x_error']:.4f},{m['obs_descend_y_error']:.4f},{m['obs_descend_z_error']:.4f})"
            )

        if self.escape_active != self.prev_escape_active:
            label = {
                0: "OFF",
                1: "ESCAPE_TRIGGER",
                2: "RECOVERY_ACTIVE",
            }.get(self.escape_active, str(self.escape_active))
            print(
                f"[Recovery] step={self.step_count} state={label} "
                f"action_move={action.get('move', action.get('ee_target_pos', []))} ee={sim_state['ee_pos'].tolist()}"
            )

        if self.control_mode == "active_inference" and self.current_belief is not None:
            release_warn = int(self.current_belief.get("release_warning", 0))
            if release_warn == 1 and self.prev_ai_release_warning == 0:
                release_counter = int(self.current_belief.get("release_contact_counter", 0))
                release_warn_steps = int(self.active_inference_params.get("release_contact_warn_steps", 0))
                print(
                    f"[AI-ReleaseWarn] step={self.step_count} phase={phase} "
                    f"contact_hold={release_counter}/{release_warn_steps}"
                )
            singularity_warn = int(self.ai_risk_state.get("singularity_warn", 0))
            if singularity_warn == 1 and self.prev_ai_singularity_warn == 0:
                print(
                    f"[AI-Risk] step={self.step_count} type=singularity_like "
                    f"phase={phase} dq_ratio={float(self.ai_risk_state.get('dq_ratio', 0.0)):.2f} "
                    f"stall={int(self.ai_risk_state.get('phase_no_progress_steps', 0))} "
                    f"counter={int(self.ai_risk_state.get('singularity_counter', 0))}/"
                    f"{self.ai_singularity_no_progress_steps}"
                )
            unintended_warn = int(self.ai_risk_state.get("unintended_contact_warn", 0))
            if unintended_warn == 1 and self.prev_ai_unintended_contact_warn == 0:
                print(
                    f"[AI-Risk] step={self.step_count} type=unexpected_contact "
                    f"phase={phase} counter={int(self.ai_risk_state.get('unintended_contact_counter', 0))}/"
                    f"{self.ai_unintended_contact_warn_steps}"
                )
            self.prev_ai_release_warning = release_warn
            self.prev_ai_singularity_warn = singularity_warn
            self.prev_ai_unintended_contact_warn = unintended_warn
        else:
            self.prev_ai_release_warning = 0
            self.prev_ai_singularity_warn = 0
            self.prev_ai_unintended_contact_warn = 0

        self.prev_phase = phase
        self.prev_contact = contact
        self.prev_escape_active = self.escape_active
        self.prev_action_grip = grip_cmd
