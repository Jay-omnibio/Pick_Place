import time

# Simulator
from env.simulator import MujocoSimulator

# Controller
from control.controller import EEController

# Agent
from agent.agent_loop import ActiveInferenceAgent


def main():
    # ------------------------------------------------
    # Paths
    # ------------------------------------------------
    MODEL_XML_PATH = "assets/pick_and_place.xml"
    SENSOR_CONFIG_PATH = "config/sensor_config.yaml"

    # ------------------------------------------------
    # Create simulator
    # ------------------------------------------------
    simulator = MujocoSimulator(
        model_path=MODEL_XML_PATH,
        render=True
    )

    # ------------------------------------------------
    # Create controller
    # ------------------------------------------------
    controller = EEController(simulator)

    # ------------------------------------------------
    # Create Active Inference agent
    # ------------------------------------------------
    agent = ActiveInferenceAgent(
        simulator=simulator,
        controller=controller,
        sensor_config_path=SENSOR_CONFIG_PATH
    )

    # ------------------------------------------------
    # Run loop
    # ------------------------------------------------
    print("Starting Active Inference Pick-and-Place demo...")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            agent.step()
            simulator.step()
            time.sleep(0.02)  # ~50 Hz

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")


if __name__ == "__main__":
    main()
