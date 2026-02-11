import csv
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

    def __init__(self, simulator, controller, sensor_config_path, log_every_steps=20, logs_dir="logs"):
        self.simulator = simulator
        self.controller = controller
        self.log_every_steps = max(1, int(log_every_steps))
        self.step_count = 0

        with open(sensor_config_path, "r") as f:
            sensor_config = yaml.safe_load(f)

        self.sensors = SensorSuite(sensor_config)
        self.current_belief = None

        self.obs_history = deque(maxlen=50)
        self.action_history = deque(maxlen=50)

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

    def step(self):
        sim_state = self.simulator.get_state()
        observation = self.sensors.get_observation(sim_state)
        self.obs_history.append(observation)

        self.current_belief = infer_beliefs(
            observation=observation,
            previous_belief=self.current_belief,
        )

        action = select_action(self.current_belief)
        self.action_history.append(action)

        self._log_step(sim_state, observation, action)

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

