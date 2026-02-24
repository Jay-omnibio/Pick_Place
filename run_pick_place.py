import os
import time
import yaml
import argparse
import numpy as np

from env.simulator import MujocoSimulator
from control.controller import EEController
from agent.agent_loop import ActiveInferenceAgent
from backends.sensor_backend import SimSensorBackend
from backends.actuator_backend import SimActuatorBackend
from safety.safety_checker import SafetyChecker
from config.runtime_loader import load_runtime_sections


def _parse_args():
    parser = argparse.ArgumentParser(description="Run pick-place simulation.")
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Disable all inspection pauses (recommended for unattended runs/recording).",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        default=None,
        help="Open MuJoCo viewer window.",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Run headless (no viewer).",
    )
    parser.add_argument(
        "--record-video",
        type=str,
        default="",
        help="Output path for recorded video (e.g. logs/run.mp4).",
    )
    parser.add_argument(
        "--record-fps",
        type=int,
        default=25,
        help="Recorder FPS for output video.",
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=960,
        help="Recorder frame width.",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=540,
        help="Recorder frame height.",
    )
    parser.add_argument(
        "--record-every-steps",
        type=int,
        default=1,
        help="Capture one frame every N simulator steps.",
    )
    parser.add_argument(
        "--record-camera",
        type=str,
        default="watching",
        help="MuJoCo camera name for recording (default: watching).",
    )
    return parser.parse_args()


def _disable_pause_env():
    os.environ["PAUSE_ON_REACH_TO_DESCEND"] = "0"
    os.environ["PAUSE_ON_REACH_TO_DESCEND_ONCE"] = "0"
    os.environ["PAUSE_ON_PHASE_CHANGE"] = "0"
    os.environ["PAUSE_ON_GRIP_START"] = "0"


def _resolve_shared_target_pose(shared_task_cfg):
    """
    Resolve shared world target pose from common config.
    Returns (xyz or None, yaw_deg, pose6 or None).
    """
    if not isinstance(shared_task_cfg, dict) or len(shared_task_cfg) == 0:
        return None, 0.0, None

    target_xyz = np.asarray(shared_task_cfg.get("target_world_xyz", []), dtype=float).reshape(-1)
    if target_xyz.shape[0] != 3:
        raise ValueError("common.task_shared.target_world_xyz must be length-3 vector.")
    target_yaw_deg = float(shared_task_cfg.get("target_world_yaw_deg", 0.0))

    pose6 = None
    pose6_raw = shared_task_cfg.get("target_world_pose6d_deg", None)
    if pose6_raw is not None:
        pose6 = np.asarray(pose6_raw, dtype=float).reshape(-1)
        if pose6.shape[0] != 6:
            raise ValueError(
                "common.task_shared.target_world_pose6d_deg must be length-6 "
                "[x,y,z,roll_deg,pitch_deg,yaw_deg]."
            )
        if not np.all(np.isfinite(pose6)):
            raise ValueError("common.task_shared.target_world_pose6d_deg must contain finite values.")
        target_xyz = pose6[:3].copy()
        target_yaw_deg = float(pose6[5])

    return target_xyz.copy(), target_yaw_deg, (None if pose6 is None else pose6.copy())


def main():
    args = _parse_args()
    MODEL_XML_PATH = "assets/pick_and_place.xml"
    SENSOR_CONFIG_PATH = "config/sensor_config.yaml"
    SAFETY_CONFIG_PATH = "config/safety_config.yaml"
    COMMON_CONFIG_PATH = os.getenv("COMMON_CONFIG_PATH", "config/common_robot.yaml")
    FSM_CONFIG_PATH = os.getenv("FSM_CONFIG_PATH", "config/fsm_config.yaml")
    ACTIVE_INFERENCE_CONFIG_PATH = os.getenv(
        "ACTIVE_INFERENCE_CONFIG_PATH", "config/active_inference_config.yaml"
    )
    runtime_cfg = load_runtime_sections(
        common_path=COMMON_CONFIG_PATH,
        fsm_path=FSM_CONFIG_PATH,
        active_inference_path=ACTIVE_INFERENCE_CONFIG_PATH,
    )
    run_cfg = runtime_cfg["run_cfg"]
    LOG_EVERY_STEPS = int(run_cfg["log_every_steps"])
    CONTROL_MODE = str(run_cfg["control_mode"]).strip().lower()
    shared_task_cfg = runtime_cfg.get("shared_task_cfg", {})

    if args.no_pause or args.record_video:
        _disable_pause_env()

    render = True if args.render is None else bool(args.render)
    if args.record_video and args.render is None:
        # For unattended capture, default to headless unless user forces --render.
        render = False

    simulator = MujocoSimulator(
        model_path=MODEL_XML_PATH,
        render=render,
        record_path=args.record_video or None,
        record_fps=args.record_fps,
        record_width=args.record_width,
        record_height=args.record_height,
        record_every_steps=args.record_every_steps,
        record_camera=args.record_camera,
    )
    shared_target_xyz, shared_target_yaw_deg, shared_target_pose6 = _resolve_shared_target_pose(
        shared_task_cfg
    )
    shared_target_source = "common.task_shared"
    if shared_target_xyz is not None:
        simulator.set_target_position(shared_target_xyz)
    else:
        # Backward-compatible fallback when shared target is not configured.
        shared_target_xyz = np.asarray(simulator.get_target_position(), dtype=float).reshape(3)
        shared_target_yaw_deg = 0.0
        shared_target_pose6 = None
        shared_target_source = "xml_site_fallback"
    controller = EEController(simulator, config=runtime_cfg["controller_cfg"])

    with open(SENSOR_CONFIG_PATH, "r") as f:
        sensor_config = yaml.safe_load(f)
    with open(SAFETY_CONFIG_PATH, "r") as f:
        safety_config = yaml.safe_load(f)

    sensor_backend = SimSensorBackend(config_path=SENSOR_CONFIG_PATH, config=sensor_config)
    safety_checker = SafetyChecker(safety_config)
    actuator_backend = SimActuatorBackend(controller, safety_checker)

    active_cfg = dict(runtime_cfg["active_inference_cfg"])
    # Keep AI place goal aligned with shared task target.
    active_cfg["use_world_place_goal_pose"] = True
    active_cfg["place_goal_world_xyz"] = np.asarray(shared_target_xyz, dtype=float).copy()
    active_cfg["place_goal_world_yaw_deg"] = float(shared_target_yaw_deg)
    if shared_target_pose6 is not None:
        active_cfg["place_goal_world_pose6d_deg"] = np.asarray(shared_target_pose6, dtype=float).copy()
    else:
        active_cfg.pop("place_goal_world_pose6d_deg", None)

    agent = ActiveInferenceAgent(
        simulator=simulator,
        sensor_backend=sensor_backend,
        actuator_backend=actuator_backend,
        control_mode=CONTROL_MODE,
        log_every_steps=LOG_EVERY_STEPS,
        task_cfg=runtime_cfg["task_cfg"],
        policy_cfg=runtime_cfg["policy_cfg"],
        active_inference_cfg=active_cfg,
    )

    # ------------------------------------------------
    # Run loop
    # ------------------------------------------------
    print("Starting Active Inference Pick-and-Place demo...")
    print("Press Ctrl+C to stop.")
    print(f"Control mode: {CONTROL_MODE}")
    print(f"Render viewer: {int(render)}")
    if args.record_video:
        print(f"Recording: {args.record_video}")
    print(f"Inspection pauses: {int(not (args.no_pause or bool(args.record_video)))}")
    print(f"Console log frequency: every {LOG_EVERY_STEPS} steps")
    print(
        f"Runtime monitor: loop_target={agent.loop_target_ms:.1f}ms "
        f"loop_warn={agent.loop_dt_warn_ms:.1f}ms "
        f"obs_age_warn={agent.obs_age_warn_ms:.1f}ms"
    )
    print(
        f"Configs: common={runtime_cfg['common_path']} "
        f"fsm={runtime_cfg['fsm_path']} "
        f"active_inference={runtime_cfg['active_inference_path']} "
        f"(found={int(runtime_cfg['found'])}, strict={int(runtime_cfg.get('strict', False))})"
    )
    print(
        "[Config-Shared] target_world="
        f"[{shared_target_xyz[0]:+.3f},{shared_target_xyz[1]:+.3f},{shared_target_xyz[2]:+.3f}] "
        f"yaw_deg={float(shared_target_yaw_deg):.1f} "
        f"source={shared_target_source}"
    )
    for line in agent.runtime_debug_lines():
        print(line)

    try:
        while True:
            agent.step()
            if agent.is_terminal():
                print(f"Terminal phase reached: {agent.terminal_phase()}. Stopping run.")
                break
            simulator.step()
            time.sleep(0.02)  # ~50 Hz

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        agent.close()
        simulator.close()
        print(f"Saved run log: {agent.log_csv_path}")


if __name__ == "__main__":
    main()
