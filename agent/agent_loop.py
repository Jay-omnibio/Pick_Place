import numpy as np
import yaml
from collections import deque

# Sensors
from env.sensors import SensorSuite

# RxInfer model (called via Julia later)
# Placeholder interface for now
from inference_interface import infer_beliefs

# Action selection (EFE)
from inference.action_selection import select_action


class ActiveInferenceAgent:
    """
    Active Inference agent loop:
    sense → infer → act
    """

    def __init__(self, simulator, controller, sensor_config_path):
        self.simulator = simulator
        self.controller = controller

        # Load sensor config
        with open(sensor_config_path, "r") as f:
            sensor_config = yaml.safe_load(f)

        self.sensors = SensorSuite(sensor_config)

        # Rolling belief state (what EFE uses)
        self.current_belief = None

        # History buffers (optional, useful later)
        self.obs_history = deque(maxlen=50)
        self.action_history = deque(maxlen=50)

    # ------------------------------------------------
    # Main loop
    # ------------------------------------------------
    def step(self):
        """
        Execute ONE Active Inference step
        """

        # 1️⃣ Get true simulator state
        sim_state = self.simulator.get_state()

        # 2️⃣ Sense (NO ground truth leaks)
        observation = self.sensors.get_observation(sim_state)
        self.obs_history.append(observation)

        # 3️⃣ Inference (RxInfer)
        # This returns beliefs over latent states
        self.current_belief = infer_beliefs(
            observation=observation,
            previous_belief=self.current_belief
        )
        
        

        # 4️⃣ Action selection via Expected Free Energy
        action = select_action(self.current_belief)
        self.action_history.append(action)
        
        print("Phase:", self.current_belief["phase"])
        print("EE belief:", self.current_belief["s_ee_mean"])
        print("Object belief (relative):", self.current_belief["s_obj_mean"])
        print("Distance to object:", np.linalg.norm(self.current_belief["s_obj_mean"]))
        print("Chosen action:", action)
        print("-" * 40)


        # 5️⃣ Send action to controller
        self.controller.apply_action(action)

    # ------------------------------------------------
    # Run loop
    # ------------------------------------------------
    def run(self, steps=500):
        """
        Run the agent for a fixed number of steps
        """
        for _ in range(steps):
            self.step()
