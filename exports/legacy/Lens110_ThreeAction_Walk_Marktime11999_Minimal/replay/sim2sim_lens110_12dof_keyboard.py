# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lens110 lower-body 12-DoF sim2sim in MuJoCo, keyboard-controlled.

This entry point mirrors the current Lens110 lower-body velocity task (12
policy actions drive the leg joints directly, torso/arms held by PD), but the
velocity command is controlled interactively from the keyboard (same key
mapping as the original keyboard replay):

  8 / 2 : increase / decrease forward speed  vx
  4 / 6 : strafe left / right                vy
  7 / 9 : turn left / right (target heading)
  0     : reset the robot and zero commands
  F     : toggle camera follow

For the Task B three-action policy, pressing 8 up to vx=0.5 makes it start
walking; pressing 2 down to 0 makes it stop.

The Task B policy is trained with a heading command (Isaac's
``heading_command=True`` turns the third observation channel into a yaw-rate
command computed from the heading error).  This replay reproduces that
closed loop by default: keys 7/9 change the target heading and the observed
yaw-rate command is recomputed from the heading error every control step.
Pass ``--no_heading_feedback`` to disable it.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import threading
import time
import itertools
from pathlib import Path

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import mujoco
import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is only cosmetic.
    tqdm = lambda x, **_: x

try:
    from pynput import keyboard as _pynput_keyboard

    HAS_PYNPUT = True
except ImportError:  # pragma: no cover - keyboard is optional.
    HAS_PYNPUT = False

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_MJCF_CANDIDATES = [
    ROOT / "replay/mjcf/lens110_21dof.xml",
    WORKSPACE_ROOT / "frameworks/shared/lens110_isaaclab/lens110/legged_lab_lbot/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml",
    WORKSPACE_ROOT / "projects/02_dance_half_body/framework/legged_lab_upper_lower/source/legged_lab/legged_lab/data/Robots/model_humanoid_lens110/mjcf/lens110_21dof.xml",
]
DEFAULT_POLICY_ROOT = ROOT / "policy_11999"
TASK_B_POLICY_ROOT = WORKSPACE_ROOT / "projects/02_dance_half_body/framework/legged_lab_upper_lower/logs/rsl_rl/lens110_amp_leg_pitch_roll"

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

HOLD_JOINT_NAMES = [
    "torso_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.14,
    "left_hip_roll_joint": -0.01,
    "left_hip_yaw_joint": -0.10,
    "left_knee_joint": 0.36,
    "left_ankle_pitch_joint": -0.2333,
    "left_ankle_roll_joint": 0.0006,
    "right_hip_pitch_joint": -0.14,
    "right_hip_roll_joint": 0.01,
    "right_hip_yaw_joint": 0.10,
    "right_knee_joint": 0.36,
    "right_ankle_pitch_joint": -0.2333,
    "right_ankle_roll_joint": -0.0006,
    "torso_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.4,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.8,
    "right_shoulder_pitch_joint": 0.4,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.8,
}

KP = {
    ".*_hip_.*": 40.0,
    ".*_knee_joint": 40.0,
    ".*_ankle_pitch_joint": 10.0,
    ".*_ankle_roll_joint": 10.0,
    "torso_yaw_joint": 100.0,
    ".*_shoulder_.*": 20.0,
    ".*_elbow_joint": 20.0,
}

KD = {
    ".*_hip_.*": 5.0,
    ".*_knee_joint": 5.0,
    ".*_ankle_pitch_joint": 10.0,
    ".*_ankle_roll_joint": 10.0,
    "torso_yaw_joint": 5.0,
    ".*_shoulder_.*": 1.0,
    ".*_elbow_joint": 1.0,
}

TAU_LIMIT = {
    ".*_hip_pitch_joint": 80.0,
    ".*_hip_roll_joint": 80.0,
    ".*_hip_yaw_joint": 36.0,
    ".*_knee_joint": 80.0,
    ".*_ankle_.*": 36.0,
    "torso_yaw_joint": 36.0,
    ".*_shoulder_.*": 36.0,
    ".*_elbow_joint": 36.0,
}

TRAIN_DT = 0.005
TRAIN_DECIMATION = 4
ACTION_SCALE = 0.25
DEFAULT_ROOT_HEIGHT = 0.68

# --- heading feedback (sim2sim straight-walk fix) ---
# Isaac trains Task B with heading_command=True, so the yaw-rate command the
# policy observes is computed from the heading error:
#   wz_cmd = clip(heading_control_stiffness * wrap_to_pi(target_heading - yaw),
#                 -ang_vel_z_max, ang_vel_z_max)
# The original replay passed a fixed wz (0), which dropped the closed-loop
# correction the policy learned; the gait's residual yaw bias then showed up
# as a consistent rightward drift in MuJoCo.  Reproducing the training-time
# feedback (below) reduced the drift by roughly 5x in testing.
HEADING_STIFFNESS = 0.5    # == commands.base_velocity.heading_control_stiffness
HEADING_ANG_VEL_MAX = 1.0  # == commands.base_velocity.ranges.ang_vel_z
HEADING_MIN, HEADING_MAX = -np.pi, np.pi

FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
FOOT_CONTACT_GEOMS = [
    "left_ankle_roll_collision",
    "right_ankle_roll_collision",
    "left_ankle_pitch_collision",
    "right_ankle_pitch_collision",
]
TRAINING_FOOT_BOXES = {
    "left_ankle_roll_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_roll_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "left_ankle_pitch_collision": ((0.019, 0.009, -0.030), (0.0975, 0.0475, 0.010)),
    "right_ankle_pitch_collision": ((0.019, -0.009, -0.030), (0.0975, 0.0475, 0.010)),
}
SHOULDER_PITCH_GAIN = -1.6
SHOULDER_PITCH_LIMIT = 0.55
ELBOW_GAIN = 1.0
ELBOW_LIMIT = 0.25
ARM_SWING_SMOOTHING = 0.20


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


def first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    formatted = "\n".join(f"  - {path}" for path in paths)
    raise FileNotFoundError(f"No local Lens110 MJCF found. Checked:\n{formatted}")


def regex_value(table: dict[str, float], name: str) -> float:
    for pattern, value in table.items():
        if re.fullmatch(pattern, name):
            return float(value)
    raise KeyError(f"No value configured for joint {name}")


def name_to_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing {obj_type.name}: {name}")
    return obj_id


class JointSet:
    def __init__(self, model: mujoco.MjModel, names: list[str]):
        self.names = names
        self.joint_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.qvel_ids = model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.actuator_ids = self._find_actuators(model)

    def _find_actuators(self, model: mujoco.MjModel) -> np.ndarray:
        actuator_ids = np.full(len(self.names), -1, dtype=np.int32)
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            matches = np.where(self.joint_ids == joint_id)[0]
            if len(matches):
                actuator_ids[matches[0]] = actuator_id
        if np.any(actuator_ids < 0):
            missing = [name for name, actuator_id in zip(self.names, actuator_ids, strict=True) if actuator_id < 0]
            raise ValueError(f"Missing actuators for joints: {missing}")
        return actuator_ids

    def qpos(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_ids].copy()

    def qvel(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.qvel_ids].copy()


class SensorIMU:
    def __init__(self, model: mujoco.MjModel):
        self.orientation = self._sensor(model, "orientation", 4)
        self.gyro = self._sensor(model, "angular-velocity", 3)
        self.lin_vel = self._optional_sensor(model, "linear-velocity", 3)

    @staticmethod
    def _sensor(model: mujoco.MjModel, name: str, dim: int) -> tuple[int, int]:
        sensor_id = name_to_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        actual_dim = int(model.sensor_dim[sensor_id])
        if actual_dim < dim:
            raise ValueError(f"Sensor {name} dim={actual_dim}, expected at least {dim}")
        return int(model.sensor_adr[sensor_id]), dim

    @staticmethod
    def _optional_sensor(model: mujoco.MjModel, name: str, dim: int) -> tuple[int, int] | None:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or int(model.sensor_dim[sensor_id]) < dim:
            return None
        return int(model.sensor_adr[sensor_id]), dim

    def quat(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.orientation
        quat = data.sensordata[adr : adr + dim].copy()
        return quat / max(np.linalg.norm(quat), 1.0e-12)

    def gyro_body(self, data: mujoco.MjData) -> np.ndarray:
        adr, dim = self.gyro
        return data.sensordata[adr : adr + dim].copy()

    def linear_velocity_body(self, data: mujoco.MjData) -> np.ndarray:
        # MuJoCo velocimeter is expressed in the world frame; the policy
        # expects body-frame base velocity (same as Isaac's base_lin_vel_b).
        return quat_to_rotmat_wxyz(self.quat(data)).T @ data.qvel[:3]


def quat_to_rotmat_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quat(quat: np.ndarray) -> float:
    """World-frame yaw of a wxyz quaternion (same convention as Isaac's heading_w)."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def joint_ranges(model: mujoco.MjModel, joints: JointSet) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(len(joints.names), -np.inf, dtype=np.float64)
    upper = np.full(len(joints.names), np.inf, dtype=np.float64)
    for i, joint_id in enumerate(joints.joint_ids):
        if model.jnt_limited[joint_id]:
            lower[i], upper[i] = model.jnt_range[joint_id]
    return lower, upper


def patch_model(model: mujoco.MjModel, controlled: JointSet) -> None:
    model.opt.timestep = TRAIN_DT
    for name, (pos, size) in TRAINING_FOOT_BOXES.items():
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_pos[geom_id] = np.asarray(pos, dtype=np.float64)
            model.geom_size[geom_id] = np.asarray(size, dtype=np.float64)
            model.geom_type[geom_id] = mujoco.mjtGeom.mjGEOM_BOX

    foot_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_CONTACT_GEOMS
    }
    foot_geom_ids.discard(-1)
    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        active = geom_name in {"floor", "ground", "plane"} or geom_id in foot_geom_ids
        model.geom_contype[geom_id] = int(active)
        model.geom_conaffinity[geom_id] = int(active)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id] = [1.0, 0.08, 0.004]

    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof_id = model.jnt_dofadr[joint_id]
        model.dof_damping[dof_id] = 0.0
        model.dof_frictionloss[dof_id] = 0.1
        model.dof_armature[dof_id] = 0.01

    for i, actuator_id in enumerate(controlled.actuator_ids):
        kp = regex_value(KP, controlled.names[i])
        kd = regex_value(KD, controlled.names[i])
        limit = regex_value(TAU_LIMIT, controlled.names[i])
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_ctrllimited[actuator_id] = 0
        model.actuator_forcelimited[actuator_id] = 1
        model.actuator_forcerange[actuator_id] = [-limit, limit]


def geom_box_zmin(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    pos = data.geom_xpos[geom_id]
    mat = data.geom_xmat[geom_id].reshape(3, 3)
    size = model.geom_size[geom_id]
    zmin = np.inf
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = pos + mat @ (size * np.array([sx, sy, sz], dtype=np.float64))
                zmin = min(zmin, float(corner[2]))
    return zmin


def settle_base_height(model: mujoco.MjModel, data: mujoco.MjData, clearance: float = 0.002) -> None:
    zmins = []
    for name in FOOT_CONTACT_GEOMS:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            zmins.append(geom_box_zmin(model, data, geom_id))
    if zmins:
        data.qpos[2] += clearance - min(zmins)
        mujoco.mj_forward(model, data)


def set_default_pose(model: mujoco.MjModel, data: mujoco.MjData, all_joints: JointSet, root_height: float) -> None:
    data.qpos[:] = 0.0
    if model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[:3] = [0.0, 0.0, root_height]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, qpos_id in zip(all_joints.names, all_joints.qpos_ids, strict=True):
        data.qpos[qpos_id] = DEFAULT_JOINT_POS[name]
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    settle_base_height(model, data)


def latest_policy(policy_root: Path) -> Path:
    candidates = []
    for root in (policy_root, TASK_B_POLICY_ROOT):
        if not root.exists():
            continue
        if root.is_file() and root.name == "policy.pt":
            candidates.append(root)
        else:
            candidates.extend(p for p in root.glob("policy.pt") if p.is_file())
            candidates.extend(p for p in root.glob("*/exported/policy.pt") if p.is_file())
    if not candidates:
        raise FileNotFoundError(
            "No local 12-DoF leg-pitch-roll policy.pt found. Checked:\n"
            f"  - {policy_root}\n"
            f"  - {TASK_B_POLICY_ROOT}"
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


class TorchScriptPolicy:
    def __init__(self, path: Path, device: torch.device):
        self.model = torch.jit.load(str(path), map_location=device)
        self.model.eval()
        self.device = device
        # Auto-detect the policy observation dimension (45 for the original
        # framework without base_lin_vel, 48 for the Task B / sway-fix runs).
        self.obs_dim = None
        for n in range(40, 101):
            try:
                with torch.inference_mode():
                    self.model(torch.zeros(1, n, device=device))
                self.obs_dim = n
                break
            except Exception:
                continue
        if self.obs_dim is None:
            raise ValueError(f"Could not infer policy input dim for {path}")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"Obs dim {obs.shape[-1]} != policy input dim {self.obs_dim}")
        obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).to(self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.inference_mode():
            action = self.model(obs_t)
        return action.squeeze(0).detach().cpu().numpy()


def build_observation(
    data: mujoco.MjData,
    leg_joints: JointSet,
    imu: SensorIMU,
    default_leg_pos: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
    include_base_lin_vel: bool = True,
) -> np.ndarray:
    rot = quat_to_rotmat_wxyz(imu.quat(data))
    projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    obs_parts = [
        imu.gyro_body(data).astype(np.float32),
        projected_gravity.astype(np.float32),
        command.astype(np.float32),
        (leg_joints.qpos(data) - default_leg_pos).astype(np.float32),
        leg_joints.qvel(data).astype(np.float32),
        last_action.astype(np.float32),
    ]
    if include_base_lin_vel:
        obs_parts.append(imu.linear_velocity_body(data).astype(np.float32))
    return np.concatenate(obs_parts).astype(np.float32)


class ScriptedArmSwing:
    def __init__(self, model: mujoco.MjModel, hold_joint_names: list[str], default_hold_pos: np.ndarray):
        self.body_ids = np.array([name_to_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in FOOT_BODY_NAMES])
        self.root_body_id = 1
        self.local_id = {name: i for i, name in enumerate(hold_joint_names)}
        self.default = default_hold_pos.copy()
        self.target = default_hold_pos.copy()

    def reset(self) -> None:
        self.target = self.default.copy()

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        root_pos = data.xpos[self.root_body_id]
        root_mat = data.xmat[self.root_body_id].reshape(3, 3)
        foot_pos_b = (data.xpos[self.body_ids] - root_pos) @ root_mat
        foot_center_x = 0.5 * (foot_pos_b[0, 0] + foot_pos_b[1, 0])
        left_foot_swing = foot_pos_b[0, 0] - foot_center_x
        right_foot_swing = foot_pos_b[1, 0] - foot_center_x

        target = self.default.copy()
        left_arm_pitch = np.clip(
            SHOULDER_PITCH_GAIN * right_foot_swing,
            -SHOULDER_PITCH_LIMIT,
            SHOULDER_PITCH_LIMIT,
        )
        right_arm_pitch = np.clip(
            SHOULDER_PITCH_GAIN * left_foot_swing,
            -SHOULDER_PITCH_LIMIT,
            SHOULDER_PITCH_LIMIT,
        )
        left_elbow = np.clip(ELBOW_GAIN * abs(right_foot_swing), 0.0, ELBOW_LIMIT)
        right_elbow = np.clip(ELBOW_GAIN * abs(left_foot_swing), 0.0, ELBOW_LIMIT)
        target[self.local_id["left_shoulder_pitch_joint"]] += left_arm_pitch
        target[self.local_id["right_shoulder_pitch_joint"]] += right_arm_pitch
        target[self.local_id["left_elbow_joint"]] += left_elbow
        target[self.local_id["right_elbow_joint"]] += right_elbow
        self.target = ARM_SWING_SMOOTHING * target + (1.0 - ARM_SWING_SMOOTHING) * self.target
        return self.target.copy()


def pd_torque(joints: JointSet, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
    q = joints.qpos(data)
    dq = joints.qvel(data)
    kp = np.array([regex_value(KP, name) for name in joints.names], dtype=np.float64)
    kd = np.array([regex_value(KD, name) for name in joints.names], dtype=np.float64)
    limit = np.array([regex_value(TAU_LIMIT, name) for name in joints.names], dtype=np.float64)
    return np.clip(kp * (target - q) - kd * dq, -limit, limit)


def viewer_key_callback(kb: KeyboardCommand):
    """Map MuJoCo viewer key codes to the interactive command."""

    def _cb(keycode: int) -> None:
        c = chr(keycode).lower() if 32 <= keycode < 127 else ""
        if c == "8":
            kb.update_vx(kb.vx_inc)
        elif c == "2":
            kb.update_vx(-kb.vx_inc)
        elif c == "4":
            kb.update_vy(kb.vy_inc)
        elif c == "6":
            kb.update_vy(-kb.vy_inc)
        elif c == "7":
            kb.update_heading(kb.heading_inc)
        elif c == "9":
            kb.update_heading(-kb.heading_inc)
        elif c == "0":
            kb.request_reset()
            print("[KEYBOARD] reset requested")
        elif c == "q" or keycode == 256:  # Q or ESC
            kb.request_quit()
            print("[KEYBOARD] quit requested")

    return _cb


def init_viewer(args: argparse.Namespace, model: mujoco.MjModel, data: mujoco.MjData, kb: KeyboardCommand | None = None):
    if args.no_render:
        return None, None, None, None
    if not args.headless:
        from mujoco import viewer

        handle = viewer.launch_passive(
            model,
            data,
            key_callback=viewer_key_callback(kb) if kb is not None else None,
        )
        handle.cam.distance = 3.5
        handle.cam.azimuth = 90.0
        handle.cam.elevation = -18.0
        handle.cam.lookat = [0.0, 0.0, 0.75]
        return None, None, None, handle

    import cv2

    model.vis.global_.offwidth = args.viewer_width
    model.vis.global_.offheight = args.viewer_height
    renderer = mujoco.Renderer(model, width=args.viewer_width, height=args.viewer_height)
    camera = mujoco.MjvCamera()
    camera.distance = 3.5
    camera.azimuth = 90.0
    camera.elevation = -18.0
    camera.lookat = [0.0, 0.0, 0.75]
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        1.0 / (TRAIN_DT * TRAIN_DECIMATION),
        (args.viewer_width, args.viewer_height),
    )
    return renderer, camera, writer, None


def render(args: argparse.Namespace, data: mujoco.MjData, renderer, camera, writer, viewer) -> bool:
    if args.no_render:
        return True
    if args.headless:
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        writer.write(frame[:, :, ::-1])
        return True
    if not viewer.is_running():
        return False
    viewer.sync()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens110 12-DoF lower-body sim2sim.")
    parser.add_argument("--policy", type=Path, default=None, help="TorchScript policy.pt. Defaults to latest leg-pitch-roll export.")
    parser.add_argument("--mjcf", type=Path, default=None, help="Direct-ankle Lens110 MJCF.")
    parser.add_argument("--cmd_vel", type=float, nargs=3, default=[0.4, 0.0, 0.0], help="vx vy heading_target (3rd = target heading in rad).")
    parser.add_argument("--duration", type=float, default=30.0, help="Simulation duration in seconds.")
    parser.add_argument(
        "--forever",
        action="store_true",
        help="Run indefinitely (ignores --duration) until Q/ESC is pressed or the viewer is closed.",
    )
    parser.add_argument("--stand_warmup", type=float, default=0.5, help="Hold default pose before policy starts.")
    parser.add_argument("--command_ramp", type=float, default=1.0, help="Seconds to ramp command after warmup.")
    parser.add_argument("--root_height", type=float, default=DEFAULT_ROOT_HEIGHT, help="Initial root height.")
    parser.add_argument("--device", default="auto", help="cuda, cpu, or auto.")
    parser.add_argument("--print_every", type=int, default=25, help="Print every N control steps.")
    parser.add_argument("--hold_default", action="store_true", help="Run PD hold without policy actions.")
    parser.add_argument("--headless", action="store_true", help="Render to mp4 using EGL.")
    parser.add_argument("--no_render", action="store_true", help="Run physics only.")
    parser.add_argument("--output", type=Path, default=Path("/tmp/lens110_12dof_sim2sim.mp4"))
    parser.add_argument("--no_keyboard", action="store_true", help="Disable keyboard control (fixed --cmd_vel).")
    parser.add_argument("--pynput", action="store_true", help="Also start a global pynput keyboard listener (needs X11).")
    parser.add_argument("--vx_increment", type=float, default=0.1, help="vx step per key press.")
    parser.add_argument("--vy_increment", type=float, default=0.1, help="vy step per key press.")
    parser.add_argument("--dyaw_increment", type=float, default=0.1, help="heading target step per key press (rad).")
    parser.add_argument(
        "--no_heading_feedback",
        action="store_true",
        help="Disable heading-error feedback (wz_cmd = raw 3rd command, as before).",
    )
    parser.add_argument("--viewer_width", type=int, default=1280)
    parser.add_argument("--viewer_height", type=int, default=720)
    return parser.parse_args()


class KeyboardCommand:
    """Thread-safe interactive command (vx, vy, target heading)."""

    VX_MIN, VX_MAX = -1.0, 1.5
    VY_MIN, VY_MAX = -1.0, 1.0
    HDG_MIN, HDG_MAX = HEADING_MIN, HEADING_MAX

    def __init__(self, vx0: float, vy0: float, heading0: float, vx_inc: float, vy_inc: float, heading_inc: float):
        self._lock = threading.Lock()
        self.vx = float(vx0)
        self.vy = float(vy0)
        self.heading = float(heading0)
        self.vx_inc = vx_inc
        self.vy_inc = vy_inc
        self.heading_inc = heading_inc
        self.reset_requested = False
        self.quit_requested = False

    def update_vx(self, d: float) -> None:
        with self._lock:
            self.vx = min(max(self.vx + d, self.VX_MIN), self.VX_MAX)

    def update_vy(self, d: float) -> None:
        with self._lock:
            self.vy = min(max(self.vy + d, self.VY_MIN), self.VY_MAX)

    def update_heading(self, d: float) -> None:
        with self._lock:
            self.heading = min(max(self.heading + d, self.HDG_MIN), self.HDG_MAX)

    def set_heading(self, heading: float) -> None:
        with self._lock:
            self.heading = float(heading)

    def request_reset(self) -> None:
        with self._lock:
            self.reset_requested = True

    def consume_reset(self) -> bool:
        with self._lock:
            req = self.reset_requested
            self.reset_requested = False
            return req

    def request_quit(self) -> None:
        with self._lock:
            self.quit_requested = True

    def consume_quit(self) -> bool:
        with self._lock:
            req = self.quit_requested
            self.quit_requested = False
            return req

    def zero(self) -> None:
        with self._lock:
            self.vx, self.vy, self.heading = 0.0, 0.0, 0.0

    def get(self) -> np.ndarray:
        with self._lock:
            return np.asarray([self.vx, self.vy, self.heading], dtype=np.float32)


def _on_press(key, kb: KeyboardCommand) -> None:
    try:
        if not hasattr(key, "char") or key.char is None:
            return
        c = key.char.lower()
        if c == "8":
            kb.update_vx(kb.vx_inc)
        elif c == "2":
            kb.update_vx(-kb.vx_inc)
        elif c == "4":
            kb.update_vy(kb.vy_inc)
        elif c == "6":
            kb.update_vy(-kb.vy_inc)
        elif c == "7":
            kb.update_heading(kb.heading_inc)
        elif c == "9":
            kb.update_heading(-kb.heading_inc)
        elif c == "0":
            kb.request_reset()
            print("[KEYBOARD] reset requested")
        elif c == "q":
            kb.request_quit()
            print("[KEYBOARD] quit requested")
    except AttributeError:
        pass


def main() -> None:
    args = parse_args()
    mjcf_path = resolve_path(args.mjcf) if args.mjcf is not None else first_existing(DEFAULT_MJCF_CANDIDATES)
    policy_path = resolve_path(args.policy) if args.policy is not None else latest_policy(DEFAULT_POLICY_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)
    leg_joints = JointSet(model, LEG_JOINT_NAMES)
    all_controlled = JointSet(model, LEG_JOINT_NAMES + HOLD_JOINT_NAMES)
    patch_model(model, all_controlled)
    lower, upper = joint_ranges(model, leg_joints)
    set_default_pose(model, data, all_controlled, args.root_height)

    imu = SensorIMU(model)
    policy = TorchScriptPolicy(policy_path, device)
    include_base_lin_vel = policy.obs_dim == 48
    if policy.obs_dim not in (45, 48):
        raise ValueError(
            f"Policy input dim {policy.obs_dim} not supported (expected 45 or 48)"
        )

    default_leg_pos = np.array([DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES], dtype=np.float64)
    default_hold_pos = np.array([DEFAULT_JOINT_POS[name] for name in HOLD_JOINT_NAMES], dtype=np.float64)
    command = np.asarray(args.cmd_vel, dtype=np.float32)  # [vx, vy, heading_target]
    heading_feedback = not args.no_heading_feedback
    use_keyboard = not args.no_keyboard
    kb = None
    if use_keyboard:
        kb = KeyboardCommand(*command, args.vx_increment, args.vy_increment, args.dyaw_increment)
        if args.pynput and HAS_PYNPUT:
            _pynput_keyboard.Listener(on_press=lambda k: _on_press(k, kb)).start()
        print("=" * 60)
        print("键盘控制（MuJoCo 视口聚焦时生效）：")
        print("  8 / 2 : 增加 / 减小前进速度 vx")
        print("  4 / 6 : 左移 / 右移 vy")
        print("  7 / 9 : 左转 / 右转（改变目标航向）")
        print("  0     : 复位机器人并清零指令")
        print("  q/ESC : 退出播放器")
        print("  说明   : Task B 策略 vx=0.5 起步行走，vx=0 停止")
        if heading_feedback:
            print("  航向闭环: 已开启（wz_cmd 由航向误差实时计算）")
        else:
            print("  航向闭环: 已关闭（--no_heading_feedback）")
        print("=" * 60)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    initial_yaw = yaw_from_quat(imu.quat(data))
    heading_target = float(command[2])
    last_action = np.zeros(len(LEG_JOINT_NAMES), dtype=np.float32)
    arm_swing = ScriptedArmSwing(model, HOLD_JOINT_NAMES, default_hold_pos)

    renderer, camera, writer, viewer = init_viewer(args, model, data, kb)

    print(f"[INFO] policy: {policy_path}")
    print(f"[INFO] MJCF: {mjcf_path}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] controlled policy joints: {len(LEG_JOINT_NAMES)}")
    print(f"[INFO] held upper-body joints: {len(HOLD_JOINT_NAMES)}")
    print(f"[INFO] physics dt={TRAIN_DT:.4f}s, control dt={TRAIN_DT * TRAIN_DECIMATION:.4f}s")
    print(
        f"[INFO] command: (vx={command[0]:+.2f}, vy={command[1]:+.2f}, "
        f"hdg_target={command[2]:+.2f} rad) heading_feedback={heading_feedback}"
    )

    steps = None if args.forever else int(args.duration / TRAIN_DT)
    warmup_steps = int(round(args.stand_warmup / TRAIN_DT))
    ramp_steps = int(round(args.command_ramp / TRAIN_DT))
    leg_target = default_leg_pos.copy()
    hold_target = default_hold_pos.copy()
    min_root_z = float("inf")
    max_abs_yaw_rate = 0.0
    net_yaw = 0.0
    start = time.time()

    if steps is None:
        # --forever: run until Q/ESC or the viewer is closed (no 100% stop).
        step_iter = tqdm(itertools.count(), desc="Simulating (forever)")
    else:
        step_iter = tqdm(range(steps), desc="Simulating")
    for step in step_iter:
        if step % TRAIN_DECIMATION == 0:
            enabled = (not args.hold_default) and step >= warmup_steps
            if enabled and ramp_steps > 0:
                alpha = float(np.clip((step - warmup_steps) / ramp_steps, 0.0, 1.0))
            else:
                alpha = 1.0 if enabled else 0.0
            if kb is not None:
                command[:] = kb.get()
                heading_target = float(command[2])
                if args.forever and kb.consume_quit():
                    print("[KEYBOARD] quitting...")
                    break
                if kb.consume_reset():
                    data.qpos[:] = initial_qpos
                    data.qvel[:] = initial_qvel
                    kb.zero()
                    kb.set_heading(initial_yaw)
                    command[:] = kb.get()
                    heading_target = float(command[2])
                    last_action.fill(0.0)
                    leg_target = default_leg_pos.copy()
                    hold_target = default_hold_pos.copy()
                    arm_swing.reset()
            if heading_feedback:
                yaw = yaw_from_quat(imu.quat(data))
                heading_err = wrap_to_pi(heading_target - yaw)
                command[2] = float(
                    np.clip(HEADING_STIFFNESS * heading_err, -HEADING_ANG_VEL_MAX, HEADING_ANG_VEL_MAX)
                )
            obs_command = command * alpha
            obs = build_observation(
                data,
                leg_joints,
                imu,
                default_leg_pos,
                last_action,
                obs_command,
                include_base_lin_vel=include_base_lin_vel,
            )

            if enabled:
                action = policy(obs).reshape(-1)
                if action.shape[0] != len(LEG_JOINT_NAMES):
                    raise ValueError(f"Policy output dim {action.shape[0]} != 12")
                last_action = action.astype(np.float32)
                leg_target = default_leg_pos + ACTION_SCALE * last_action
                np.clip(leg_target, lower, upper, out=leg_target)
            else:
                last_action.fill(0.0)
                leg_target = default_leg_pos.copy()
                arm_swing.reset()
            hold_target = arm_swing(data) if enabled else default_hold_pos.copy()

            policy_step = step // TRAIN_DECIMATION
            if policy_step % max(args.print_every, 1) == 0:
                lin_vel = imu.linear_velocity_body(data)
                gyro = imu.gyro_body(data)
                print(
                    f"t={data.time:5.2f}s "
                    f"cmd=({obs_command[0]:+.2f},{obs_command[1]:+.2f},{obs_command[2]:+.2f}) "
                    f"hdg={heading_target:+.2f} "
                    f"vel=({lin_vel[0]:+.2f},{lin_vel[1]:+.2f},{gyro[2]:+.2f}) "
                    f"z={data.qpos[2]:+.3f} "
                    f"act_abs_max={np.max(np.abs(last_action)):.3f}"
                )

            if not render(args, data, renderer, camera, writer, viewer):
                break

        full_target = np.concatenate([leg_target, hold_target])
        data.ctrl[all_controlled.actuator_ids] = full_target
        mujoco.mj_step(model, data)
        min_root_z = min(min_root_z, float(data.qpos[2]))
        wz = float(imu.gyro_body(data)[2])
        max_abs_yaw_rate = max(max_abs_yaw_rate, abs(wz))
        net_yaw += wz * TRAIN_DT

        if not args.headless and not args.no_render:
            target_time = (step + 1) * TRAIN_DT
            elapsed = time.time() - start
            if elapsed < target_time:
                time.sleep(target_time - elapsed)

    print(
        f"[SUMMARY] min_root_z={min_root_z:.3f}, max_abs_yaw_rate={max_abs_yaw_rate:.3f}, "
        f"net_yaw={np.degrees(net_yaw):+.1f}deg over {data.time:.0f}s"
    )
    if writer is not None:
        writer.release()
        print(f"[INFO] saved video: {args.output}")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
