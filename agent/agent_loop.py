import csv
import os
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from env.sensors import SensorSuite
from inference_interface import infer_beliefs
from inference.action_selection import select_action

try:
    from inference_interface import REACH_OBJ_REL
except Exception:
    REACH_OBJ_REL = np.array([0.0, 0.0, -0.08])


class ActiveInferenceAgent:
    """
    Active Inference agent loop:
    sense -> infer -> act
    """

    def __init__(self, simulator, controller, sensor_config_path, log_every_steps=100, logs_dir="logs"):
        self.simulator = simulator
        self.controller = controller
        self.log_every_steps = max(1, int(log_every_steps))
        self.step_count = 0

        with open(sensor_config_path, "r") as f:
            sensor_config = yaml.safe_load(f)

        self.sensors = SensorSuite(sensor_config)
        self.current_belief = None
        self.prev_phase = None
        self.prev_grasp = None
        self.prev_contact = None
        self.prev_escape_active = 0
        self.log_contact_events = os.getenv("LOG_CONTACT_EVENTS", "0") == "1"

        self.obs_history = deque(maxlen=50)
        self.action_history = deque(maxlen=50)
        self.ee_true_history = deque(maxlen=120)
        self.escape_cooldown = 0
        self.escape_active = 0
        self.stall_window = 70
        self.stall_disp_threshold = 0.003
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
            "s_ee_x",
            "s_ee_y",
            "s_ee_z",
            "s_obj_rel_x",
            "s_obj_rel_y",
            "s_obj_rel_z",
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
        ]

    def _log_step(self, sim_state, observation, action):
        ee = np.asarray(self.current_belief["s_ee_mean"], dtype=float)
        obj_rel = np.asarray(self.current_belief["s_obj_mean"], dtype=float)
        move = np.asarray(action["move"], dtype=float)

        row = {
            "step": self.step_count,
            "phase": self.current_belief["phase"],
            "distance_to_object": float(np.linalg.norm(obj_rel)),
            "reach_error": float(np.linalg.norm(obj_rel - REACH_OBJ_REL)),
            "s_ee_x": float(ee[0]),
            "s_ee_y": float(ee[1]),
            "s_ee_z": float(ee[2]),
            "s_obj_rel_x": float(obj_rel[0]),
            "s_obj_rel_y": float(obj_rel[1]),
            "s_obj_rel_z": float(obj_rel[2]),
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
        }
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
        if self.current_belief is None or self.current_belief.get("phase") != "Reach":
            self.escape_cooldown = max(0, self.escape_cooldown - 1)
            self.recovery_steps_remaining = 0
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
        obj_rel = np.asarray(self.current_belief.get("s_obj_mean", [0.0, 0.0, 0.0]), dtype=float)
        reach_error = float(np.linalg.norm(obj_rel - REACH_OBJ_REL))

        if displacement >= self.stall_disp_threshold or reach_error < 0.15:
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

    def step(self):
        sim_state = self.simulator.get_state()
        self.ee_true_history.append(np.array(sim_state["ee_pos"], dtype=float))
        observation = self.sensors.get_observation(sim_state)
        self.obs_history.append(observation)

        self.current_belief = infer_beliefs(
            observation=observation,
            previous_belief=self.current_belief,
        )

        action = select_action(self.current_belief)
        action = self._maybe_escape_stall(action, sim_state)
        self.action_history.append(action)

        self._log_step(sim_state, observation, action)
        self._log_events(sim_state, observation, action)

        if self.step_count % self.log_every_steps == 0:
            obj_rel = np.asarray(self.current_belief["s_obj_mean"], dtype=float)
            print(f"Step: {self.step_count}")
            print("Phase:", self.current_belief["phase"])
            print("EE belief:", self.current_belief["s_ee_mean"])
            print("Object belief (relative):", obj_rel)
            print("Distance to object:", float(np.linalg.norm(obj_rel)))
            print("Reach error:", float(np.linalg.norm(obj_rel - REACH_OBJ_REL)))
            print("Chosen action:", action)
            print("-" * 40)

        self.controller.apply_action(action)
        self.step_count += 1

    def close(self):
        if hasattr(self, "_csv_file") and self._csv_file and not self._csv_file.closed:
            self._csv_file.flush()
            self._csv_file.close()

    def _log_events(self, sim_state, observation, action):
        """
        Event logs for key state/action transitions so run behavior is easier to inspect.
        """
        phase = self.current_belief.get("phase")
        grasp = int(self.current_belief.get("s_grasp", 0))
        contact = int(observation.get("o_contact", 0))
        obj_rel = np.asarray(self.current_belief.get("s_obj_mean", [0.0, 0.0, 0.0]), dtype=float)
        reach_error = float(np.linalg.norm(obj_rel - REACH_OBJ_REL))

        if self.prev_phase is None:
            self.prev_phase = phase
            self.prev_grasp = grasp
            self.prev_contact = contact
            self.prev_escape_active = self.escape_active
            print(f"[Init] phase={phase} grasp={grasp} contact={contact}")
            return

        if phase != self.prev_phase:
            print(
                f"[Phase] step={self.step_count} {self.prev_phase} -> {phase} "
                f"reach_error={reach_error:.4f} contact={contact} grasp={grasp}"
            )

        if grasp != self.prev_grasp:
            if grasp == 1:
                print(
                    f"[GraspAcquired] step={self.step_count} "
                    f"obj_rel={obj_rel.tolist()} grip_width={float(observation.get('o_grip', 0.0)):.4f}"
                )
            else:
                print(
                    f"[GraspLost] step={self.step_count} "
                    f"obj_rel={obj_rel.tolist()} grip_width={float(observation.get('o_grip', 0.0)):.4f}"
                )

        if contact != self.prev_contact and self.log_contact_events:
            state = "ON" if contact == 1 else "OFF"
            print(f"[Contact{state}] step={self.step_count} reach_error={reach_error:.4f}")

        if self.escape_active != self.prev_escape_active:
            label = {
                0: "OFF",
                1: "ESCAPE_TRIGGER",
                2: "RECOVERY_ACTIVE",
            }.get(self.escape_active, str(self.escape_active))
            print(
                f"[Recovery] step={self.step_count} state={label} "
                f"action_move={action['move']} ee={sim_state['ee_pos'].tolist()}"
            )

        self.prev_phase = phase
        self.prev_grasp = grasp
        self.prev_contact = contact
        self.prev_escape_active = self.escape_active
