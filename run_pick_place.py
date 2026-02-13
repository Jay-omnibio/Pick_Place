import os
import time
import yaml

from env.simulator import MujocoSimulator
from control.controller import EEController
from agent.agent_loop import ActiveInferenceAgent
from backends.sensor_backend import SimSensorBackend
from backends.actuator_backend import SimActuatorBackend
from safety.safety_checker import SafetyChecker


def main():
    MODEL_XML_PATH = "assets/pick_and_place.xml"
    SENSOR_CONFIG_PATH = "config/sensor_config.yaml"
    SAFETY_CONFIG_PATH = "config/safety_config.yaml"
    LOG_EVERY_STEPS = int(os.getenv("LOG_EVERY_STEPS", "100"))
    CONTROL_MODE = os.getenv("CONTROL_MODE", "fsm").strip().lower()

    simulator = MujocoSimulator(model_path=MODEL_XML_PATH, render=True)
    controller = EEController(simulator)

    with open(SENSOR_CONFIG_PATH, "r") as f:
        sensor_config = yaml.safe_load(f)
    with open(SAFETY_CONFIG_PATH, "r") as f:
        safety_config = yaml.safe_load(f)

    sensor_backend = SimSensorBackend(config_path=SENSOR_CONFIG_PATH, config=sensor_config)
    safety_checker = SafetyChecker(safety_config)
    actuator_backend = SimActuatorBackend(controller, safety_checker)

    agent = ActiveInferenceAgent(
        simulator=simulator,
        sensor_backend=sensor_backend,
        actuator_backend=actuator_backend,
        control_mode=CONTROL_MODE,
        log_every_steps=LOG_EVERY_STEPS,
    )

    # ------------------------------------------------
    # Run loop
    # ------------------------------------------------
    print("Starting Active Inference Pick-and-Place demo...")
    print("Press Ctrl+C to stop.")
    print(f"Control mode: {CONTROL_MODE}")
    print(f"Console log frequency: every {LOG_EVERY_STEPS} steps")

    try:
        while True:
            agent.step()
            simulator.step()
            time.sleep(0.02)  # ~50 Hz

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        agent.close()
        print(f"Saved run log: {agent.log_csv_path}")


if __name__ == "__main__":
    main()
