import argparse
import contextlib
import io
import math
import os
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import yaml

from agent.agent_loop import ActiveInferenceAgent
from backends.actuator_backend import SimActuatorBackend
from backends.sensor_backend import SimSensorBackend
from config.runtime_loader import (
    DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH,
    DEFAULT_COMMON_CONFIG_PATH,
    DEFAULT_FSM_CONFIG_PATH,
    load_runtime_sections,
)
from control.controller import EEController
from env.simulator import MujocoSimulator
from safety.safety_checker import SafetyChecker
from tasks.pick_place_fsm import Phase


def _phase_name(agent: ActiveInferenceAgent) -> str:
    if agent.task_state is None:
        return "Unknown"
    return Phase(agent.task_state.phase).value


def _set_pause_flags(disable_pauses: bool) -> None:
    if not disable_pauses:
        return
    os.environ["PAUSE_ON_REACH_TO_DESCEND"] = "0"
    os.environ["PAUSE_ON_REACH_TO_DESCEND_ONCE"] = "0"
    os.environ["PAUSE_ON_PHASE_CHANGE"] = "0"
    os.environ["PAUSE_ON_GRIP_START"] = "0"


def _wrap_to_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def _quat_wxyz_to_yaw(quat_wxyz) -> float:
    w, x, y, z = [float(v) for v in quat_wxyz]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def _yaw_deg_to_quat_wxyz(yaw_deg: float) -> np.ndarray:
    yaw = float(np.deg2rad(yaw_deg))
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)], dtype=float)


def _set_object_pose(
    simulator: MujocoSimulator,
    x: float,
    y: float,
    z: float,
    yaw_deg: float,
) -> None:
    """
    Set object free-joint pose directly:
    qpos free-joint layout: [x, y, z, qw, qx, qy, qz].
    """
    jid = int(mujoco.mj_name2id(simulator.model, mujoco.mjtObj.mjOBJ_JOINT, "obj_joint"))
    if jid < 0:
        raise ValueError("Joint 'obj_joint' not found in model.")

    qpos_adr = int(simulator.model.jnt_qposadr[jid])
    qvel_adr = int(simulator.model.jnt_dofadr[jid])
    quat = _yaw_deg_to_quat_wxyz(yaw_deg)

    simulator.data.qpos[qpos_adr : qpos_adr + 3] = np.array([x, y, z], dtype=float)
    simulator.data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    simulator.data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(simulator.model, simulator.data)


def _sample_object_pose(
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> Optional[Tuple[float, float, float, float]]:
    if args.fixed_object:
        return None
    x = float(rng.uniform(args.obj_x_min, args.obj_x_max))
    y = float(rng.uniform(args.obj_y_min, args.obj_y_max))
    z = float(args.obj_z)
    yaw_deg = float(rng.uniform(args.obj_yaw_min, args.obj_yaw_max))
    return (x, y, z, yaw_deg)


def _gripper_yaw_target_deg(init_obj_yaw_deg: float, policy_cfg: Any) -> Optional[float]:
    enable = bool(getattr(policy_cfg, "enable_object_yaw_align", False))
    if not enable:
        return None
    offset_deg = float(getattr(policy_cfg, "object_yaw_offset_deg", 90.0))
    target_rad = _wrap_to_pi(np.deg2rad(init_obj_yaw_deg + offset_deg))
    return float(np.rad2deg(target_rad))


def _run_one_episode(
    episode_idx: int,
    runtime_cfg: Dict[str, Any],
    sensor_config: Dict[str, Any],
    safety_config: Dict[str, Any],
    model_xml_path: str,
    sensor_config_path: str,
    max_steps: int,
    log_every_steps: int,
    render: bool,
    object_pose: Optional[Tuple[float, float, float, float]],
    quiet_internal_logs: bool = True,
) -> Dict[str, Any]:
    simulator = MujocoSimulator(model_path=model_xml_path, render=render)
    if object_pose is not None:
        _set_object_pose(
            simulator,
            x=float(object_pose[0]),
            y=float(object_pose[1]),
            z=float(object_pose[2]),
            yaw_deg=float(object_pose[3]),
        )

    init_state = simulator.get_state()
    init_obj = np.asarray(init_state["obj_pos"], dtype=float)
    init_obj_yaw_deg = float(np.rad2deg(_quat_wxyz_to_yaw(init_state.get("obj_quat_wxyz", [1.0, 0.0, 0.0, 0.0]))))

    controller = EEController(simulator, config=runtime_cfg["controller_cfg"])
    sensor_backend = SimSensorBackend(config_path=sensor_config_path, config=sensor_config)
    safety_checker = SafetyChecker(safety_config)
    actuator_backend = SimActuatorBackend(controller, safety_checker)

    agent = ActiveInferenceAgent(
        simulator=simulator,
        sensor_backend=sensor_backend,
        actuator_backend=actuator_backend,
        control_mode="fsm",
        log_every_steps=int(log_every_steps),
        task_cfg=runtime_cfg["task_cfg"],
        policy_cfg=runtime_cfg["policy_cfg"],
    )

    reached_lift = False
    outcome = "max_steps"
    final_phase = "Unknown"

    def _episode_loop() -> None:
        nonlocal reached_lift, outcome, final_phase
        for _ in range(max_steps):
            agent.step()
            final_phase = _phase_name(agent)

            if final_phase == Phase.LiftTest.value:
                reached_lift = True
                outcome = "lift_reached"
                break

            if agent.is_terminal():
                outcome = f"terminal_{agent.terminal_phase().lower()}"
                break

            simulator.step()

    try:
        if quiet_internal_logs:
            with contextlib.redirect_stdout(io.StringIO()):
                _episode_loop()
        else:
            _episode_loop()
    finally:
        agent.close()
        if getattr(simulator, "viewer", None) is not None and hasattr(simulator.viewer, "close"):
            simulator.viewer.close()

    yaw_target_deg = _gripper_yaw_target_deg(
        init_obj_yaw_deg=init_obj_yaw_deg,
        policy_cfg=runtime_cfg["policy_cfg"],
    )

    return {
        "episode": episode_idx,
        "reached_lift": reached_lift,
        "outcome": outcome,
        "steps": int(agent.step_count),
        "final_phase": final_phase,
        "csv": str(agent.log_csv_path),
        "init_obj_x": float(init_obj[0]),
        "init_obj_y": float(init_obj[1]),
        "init_obj_z": float(init_obj[2]),
        "init_obj_yaw_deg": init_obj_yaw_deg,
        "gripper_yaw_target_deg": yaw_target_deg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple FSM episodes and stop each episode when LiftTest is reached."
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run (recommended 5-10).")
    parser.add_argument("--max-steps", type=int, default=5000, help="Step budget per episode.")
    parser.add_argument("--log-every-steps", type=int, default=1000, help="Heartbeat print frequency.")
    parser.add_argument("--render", action="store_true", help="Render MuJoCo viewer (slower).")
    parser.add_argument(
        "--disable-pauses",
        action="store_true",
        help="Disable pause prompts so batch run is non-interactive.",
    )
    parser.add_argument(
        "--show-internal-logs",
        action="store_true",
        help="Show internal [HB]/[Phase]/[CtrlDbg] logs (default is hidden).",
    )
    parser.add_argument(
        "--common-config",
        default=os.getenv("COMMON_CONFIG_PATH", DEFAULT_COMMON_CONFIG_PATH),
        help="Path to common_robot config.",
    )
    parser.add_argument(
        "--fsm-config",
        default=os.getenv("FSM_CONFIG_PATH", DEFAULT_FSM_CONFIG_PATH),
        help="Path to fsm config.",
    )
    parser.add_argument(
        "--active-inference-config",
        default=os.getenv("ACTIVE_INFERENCE_CONFIG_PATH", DEFAULT_ACTIVE_INFERENCE_CONFIG_PATH),
        help="Path to active-inference config (required by strict loader, even in FSM batch).",
    )
    parser.add_argument("--model-xml", default="assets/pick_and_place.xml", help="MuJoCo model XML.")
    parser.add_argument("--sensor-config", default="config/sensor_config.yaml", help="Sensor config YAML.")
    parser.add_argument("--safety-config", default="config/safety_config.yaml", help="Safety config YAML.")
    parser.add_argument("--fixed-object", action="store_true", help="Keep XML object pose fixed every episode.")
    parser.add_argument("--obj-x-min", type=float, default=0.35, help="Random object X min (m).")
    parser.add_argument("--obj-x-max", type=float, default=0.60, help="Random object X max (m).")
    parser.add_argument("--obj-y-min", type=float, default=-0.15, help="Random object Y min (m).")
    parser.add_argument("--obj-y-max", type=float, default=0.15, help="Random object Y max (m).")
    parser.add_argument("--obj-z", type=float, default=0.20, help="Object Z height for sampled poses (m).")
    parser.add_argument("--obj-yaw-min", type=float, default=-90.0, help="Random object yaw min (deg).")
    parser.add_argument("--obj-yaw-max", type=float, default=90.0, help="Random object yaw max (deg).")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed for reproducible sampling.")
    args = parser.parse_args()

    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.log_every_steps < 1:
        raise ValueError("--log-every-steps must be >= 1")
    if args.obj_x_min > args.obj_x_max:
        raise ValueError("--obj-x-min must be <= --obj-x-max")
    if args.obj_y_min > args.obj_y_max:
        raise ValueError("--obj-y-min must be <= --obj-y-max")
    if args.obj_yaw_min > args.obj_yaw_max:
        raise ValueError("--obj-yaw-min must be <= --obj-yaw-max")

    _set_pause_flags(args.disable_pauses)
    rng = np.random.default_rng(int(args.seed))

    runtime_cfg = load_runtime_sections(
        common_path=args.common_config,
        fsm_path=args.fsm_config,
        active_inference_path=args.active_inference_config,
    )
    with open(args.sensor_config, "r", encoding="utf-8") as f:
        sensor_config = yaml.safe_load(f)
    with open(args.safety_config, "r", encoding="utf-8") as f:
        safety_config = yaml.safe_load(f)

    print("Starting FSM batch (stop at LiftTest)...")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Heartbeat every: {args.log_every_steps} steps")
    print(f"Render: {int(args.render)}")
    print(f"Disable pauses: {int(args.disable_pauses)}")
    print(
        "Configs: "
        f"common={args.common_config} "
        f"fsm={args.fsm_config} "
        f"active_inference={args.active_inference_config}"
    )

    results = []
    for ep in range(1, args.episodes + 1):
        object_pose = _sample_object_pose(rng=rng, args=args)
        result = _run_one_episode(
            episode_idx=ep,
            runtime_cfg=runtime_cfg,
            sensor_config=sensor_config,
            safety_config=safety_config,
            model_xml_path=args.model_xml,
            sensor_config_path=args.sensor_config,
            max_steps=int(args.max_steps),
            log_every_steps=int(args.log_every_steps),
            render=bool(args.render),
            object_pose=object_pose,
            quiet_internal_logs=not bool(args.show_internal_logs),
        )
        results.append(result)

        yaw_set = (
            "disabled"
            if result["gripper_yaw_target_deg"] is None
            else f"{result['gripper_yaw_target_deg']:.1f}deg"
        )
        done = "yes" if result["reached_lift"] else "no"
        print(
            f"[Episode {ep}] "
            f"obj=({result['init_obj_x']:.3f},{result['init_obj_y']:.3f},{result['init_obj_z']:.3f}) "
            f"obj_yaw={result['init_obj_yaw_deg']:.1f}deg "
            f"gripper_yaw_set={yaw_set} "
            f"done={done} final_phase={result['final_phase']}"
        )

    total = len(results)
    lift_ok = sum(1 for r in results if r["reached_lift"])
    lift_rate = 100.0 * lift_ok / max(1, total)
    print("-" * 72)
    print(
        f"Summary: lift_reached={lift_ok}/{total} ({lift_rate:.1f}%) "
        f"| not_reached={total - lift_ok}/{total}"
    )


if __name__ == "__main__":
    main()
