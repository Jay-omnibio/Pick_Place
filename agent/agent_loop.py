import csv
import os
import math
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

from policies.scripted_pick_place import ScriptedPickPlacePolicy
from tasks.pick_place_fsm import Phase, TaskConfig, step_fsm
from inference_interface import infer_beliefs
from inference.action_selection import select_action


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
            raise ValueError("task_cfg is required. Load it from config/tuning_config.yaml.")
        if policy_cfg is None:
            raise ValueError("policy_cfg is required. Load it from config/tuning_config.yaml.")
        self.task_cfg = task_cfg
        self.task_state = None
        self.policy = ScriptedPickPlacePolicy(task_cfg=self.task_cfg, policy_cfg=policy_cfg)
        self.current_belief = None
        self.prev_phase = None
        self.prev_contact = None
        self.prev_escape_active = 0
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
            "distance_to_object",
            "reach_error",
            "pregrasp_error",
            "descend_error",
            "descend_x_error",
            "descend_y_error",
            "descend_z_error",
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

    def _phase_name(self):
        if self.control_mode == "active_inference":
            if self.current_belief is None:
                return "Reach"
            return str(self.current_belief.get("phase", "Reach"))
        return self.task_state.phase.value if self.task_state is not None else Phase.ReachAbove.value

    def _get_references(self):
        if self.control_mode == "active_inference":
            reach_ref = np.array([0.0, 0.0, -0.08], dtype=float)
            return reach_ref, reach_ref
        return self.task_cfg.pregrasp_obj_rel, self.task_cfg.grasp_obj_rel

    def _collect_log_metrics(self, sim_state, observation):
        reach_ref, descend_ref = self._get_references()
        obj_rel = np.asarray(observation.get("o_obj", [0.0, 0.0, 0.0]), dtype=float)
        target_rel = np.asarray(observation.get("o_target", [0.0, 0.0, 0.0]), dtype=float)
        ee_pos = np.asarray(sim_state.get("ee_pos", [0.0, 0.0, 0.0]), dtype=float)
        obj_pos = np.asarray(sim_state.get("obj_pos", [0.0, 0.0, 0.0]), dtype=float)
        target_pos = np.asarray(sim_state.get("target_pos", [0.0, 0.0, 0.0]), dtype=float)
        true_obj_rel = obj_pos - ee_pos
        true_target_rel = target_pos - ee_pos

        true_reach_err_vec = true_obj_rel - reach_ref
        true_descend_err_vec = true_obj_rel - descend_ref
        true_preplace_err_vec = true_target_rel - self.task_cfg.preplace_target_rel
        true_place_err_vec = true_target_rel - self.task_cfg.place_target_rel
        obs_reach_err_vec = obj_rel - reach_ref
        obs_descend_err_vec = obj_rel - descend_ref
        obs_preplace_err_vec = target_rel - self.task_cfg.preplace_target_rel
        obs_place_err_vec = target_rel - self.task_cfg.place_target_rel

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
            "obs_preplace_error": float(np.linalg.norm(target_rel - self.task_cfg.preplace_target_rel)),
            "obs_place_error": float(np.linalg.norm(target_rel - self.task_cfg.place_target_rel)),
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
            step_in_phase=int(self.task_state.step_in_phase),
            descend_contact_hold=int(self.task_state.descend_contact_hold),
            descend_ready_counter=int(self.task_state.descend_ready_counter),
        )

    def _log_heartbeat(self, sim_state, observation, action):
        phase = self._phase_name()
        contact = int(observation.get("o_contact", 0))
        grip_cmd = int(action.get("grip", 0))
        m = self._collect_log_metrics(sim_state, observation)
        msg = f"[HB] step={self.step_count} phase={phase} contact={contact} grip={grip_cmd}"

        if phase == Phase.ReachAbove.value:
            msg += (
                f" x={m['true_reach_x_error']:.4f} "
                f"y={m['true_reach_y_error']:.4f} "
                f"xy={m['true_reach_xy_error']:.4f}/{self.task_cfg.reach_xy_threshold:.4f} "
                f"z={m['true_reach_z_error']:.4f}/{self.task_cfg.reach_z_threshold:.4f}"
            )

        elif phase == Phase.Descend.value:
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
            msg += f" ee_z={m['true_ee_z']:.4f} transit_target_z={self.task_cfg.transit_height:.4f}"
        elif phase == Phase.MoveToPlaceAbove.value:
            msg += (
                f" x={m['true_preplace_x_error']:.4f} "
                f"y={m['true_preplace_y_error']:.4f} "
                f"xy={m['true_preplace_xy_error']:.4f}/{self.task_cfg.preplace_xy_threshold:.4f} "
                f"z={m['true_preplace_z_error']:.4f}/{self.task_cfg.preplace_z_threshold:.4f} "
                f"norm={m['obs_preplace_error']:.4f}/{self.task_cfg.preplace_threshold:.4f} "
                f"ee_z={m['true_ee_z']:.4f}"
            )
        elif phase == Phase.DescendToPlace.value:
            msg += (
                f" x={m['true_place_x_error']:.4f} "
                f"y={m['true_place_y_error']:.4f} "
                f"xy={m['true_place_xy_error']:.4f}/{self.task_cfg.place_xy_threshold:.4f} "
                f"z={m['true_place_z_error']:.4f}/{self.task_cfg.place_z_threshold:.4f} "
                f"norm={m['obs_place_error']:.4f}/{self.task_cfg.place_threshold:.4f} "
                f"ee_z={m['true_ee_z']:.4f} obj_z={m['true_obj_z']:.4f}"
            )
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

        if self.log_pose_debug:
            hand_roll, hand_pitch, hand_yaw = self._quat_wxyz_to_rpy(
                sim_state.get("hand_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
            )
            joint7_pos = float(sim_state.get("joint7_pos", 0.0))
            msg += (
                f" joint7={joint7_pos:.4f} "
                f"rpy=({hand_roll:.4f},{hand_pitch:.4f},{hand_yaw:.4f})"
            )

        print(msg)

    def _log_step(self, sim_state, observation, action):
        ee = np.asarray(observation["o_ee"], dtype=float)
        obj_rel = np.asarray(observation["o_obj"], dtype=float)
        target_rel = np.asarray(observation["o_target"], dtype=float)
        if self.control_mode == "active_inference":
            pregrasp_ref = np.array([0.0, 0.0, -0.08], dtype=float)
            descend_ref = pregrasp_ref
        else:
            pregrasp_ref = self.task_cfg.pregrasp_obj_rel
            descend_ref = self.task_cfg.grasp_obj_rel
        pregrasp_error = float(np.linalg.norm(obj_rel - pregrasp_ref))
        descend_error = float(np.linalg.norm(obj_rel - descend_ref))
        descend_x_error = float(abs((obj_rel - descend_ref)[0]))
        descend_y_error = float(abs((obj_rel - descend_ref)[1]))
        descend_z_error = float(abs((obj_rel - descend_ref)[2]))
        if "ee_target_pos" in action:
            move = np.asarray(action["ee_target_pos"], dtype=float) - np.asarray(sim_state["ee_pos"], dtype=float)
        else:
            move = np.asarray(action.get("move", [0.0, 0.0, 0.0]), dtype=float)

        row = {
            "step": self.step_count,
            "phase": (
                self.current_belief.get("phase", "Reach")
                if self.control_mode == "active_inference" and self.current_belief is not None
                else (self.task_state.phase.value if self.task_state is not None else Phase.ReachAbove.value)
            ),
            "distance_to_object": float(np.linalg.norm(obj_rel)),
            "reach_error": pregrasp_error,
            "pregrasp_error": pregrasp_error,
            "descend_error": descend_error,
            "descend_x_error": descend_x_error,
            "descend_y_error": descend_y_error,
            "descend_z_error": descend_z_error,
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
        reach_ref = (
            np.array([0.0, 0.0, -0.08], dtype=float)
            if self.control_mode == "active_inference"
            else self.task_cfg.pregrasp_obj_rel
        )
        reach_error = float(np.linalg.norm(obj_rel - reach_ref))

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

    def _compute_descend_gate(self, obj_rel, step_in_phase, descend_contact_hold, descend_ready_counter):
        """
        Compute descend trigger/gate status using the same conditions as step_fsm.
        This is used only for explainability logs.
        """
        obj_rel = np.asarray(obj_rel, dtype=float)
        err = obj_rel - self.task_cfg.grasp_obj_rel
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
        contact = int(observation.get("o_contact", 0))
        err = obj_rel - self.task_cfg.grasp_obj_rel
        xy_ok = (
            abs(float(err[0])) <= self.task_cfg.descend_x_threshold
            and abs(float(err[1])) <= self.task_cfg.descend_y_threshold
        )
        z_ok = abs(float(err[2])) <= self.task_cfg.descend_z_threshold
        descend_contact_hold = (int(prev_state.descend_contact_hold) + 1) if contact == 1 else 0
        descend_ready_counter = (int(prev_state.descend_ready_counter) + 1) if (xy_ok and z_ok) else 0
        return self._compute_descend_gate(
            obj_rel=obj_rel,
            step_in_phase=int(prev_state.step_in_phase) + 1,
            descend_contact_hold=descend_contact_hold,
            descend_ready_counter=descend_ready_counter,
        )

    def step(self):
        sim_state = self.simulator.get_state()
        self.ee_true_history.append(np.array(sim_state["ee_pos"], dtype=float))
        observation = self.sensor_backend.get_observation(sim_state)
        self.obs_history.append(observation)

        if self.control_mode == "active_inference":
            self.current_belief = infer_beliefs(observation=observation, previous_belief=self.current_belief)
            action = select_action(self.current_belief)
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
        descend_ref = self.task_cfg.grasp_obj_rel if self.control_mode != "active_inference" else np.array([0.0, 0.0, -0.08], dtype=float)
        descend_x_error = float(abs((obj_rel - descend_ref)[0]))
        descend_y_error = float(abs((obj_rel - descend_ref)[1]))
        descend_z_error = float(abs((obj_rel - descend_ref)[2]))
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

        self.prev_phase = phase
        self.prev_contact = contact
        self.prev_escape_active = self.escape_active
        self.prev_action_grip = grip_cmd
