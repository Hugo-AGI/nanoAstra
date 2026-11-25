"""RoboTwin 2.0 (aloha-agilex, ``demo_randomized``) behind the step interface.

Episodes follow RoboTwin's ``script/eval_policy.py``: the seeds are ones its expert solves, the expert runs
on the task object right before the policy, the instruction is drawn from the unseen set, and success is
RoboTwin's own check.  The seeds of every task are fixed (``seeds/``), so every run sees the same scenes.
One step is one ``take_action`` in end-effector mode: a planned motion of both arms.

The model works in the robot frame (+x forward, +y to the robot's left, +z up, origin at the robot base)
and controls the grasp point between the fingers.  It sees an overview camera and the two wrist cameras,
mounted above the grippers.
"""

from __future__ import annotations

import atexit
import functools
import glob
import importlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

from .. import overlay
from ..prompt import gripper_reading
from ..steps import ArmStep, Command

logger = logging.getLogger(__name__)

ROOT = Path(os.environ.get("ROBOTWIN_ROOT") or Path(__file__).resolve().parents[2] / "RoboTwin").resolve()
if not (ROOT / "envs").is_dir():
    raise FileNotFoundError(f"RoboTwin not found at {ROOT}: run `git submodule update --init` or set ROBOTWIN_ROOT")
sys.path[:0] = [str(ROOT), str(ROOT / "description" / "utils")]
os.chdir(ROOT)                      # RoboTwin loads its assets by relative path
# RoboTwin draws object variants from glob and Path.iterdir listings, whose order depends on the file
# system; sorting them makes a seed build the same scene on every machine.
_glob, _iterdir = glob.glob, Path.iterdir
glob.glob = lambda *args, **kwargs: sorted(_glob(*args, **kwargs))
Path.iterdir = lambda self: iter(sorted(_iterdir(self)))
logging.getLogger("curobo").setLevel(logging.WARNING)

import transforms3d as t3d  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import generate_episode_descriptions  # noqa: E402

TASK_CONFIG = "demo_randomized"
SEEDS_DIR = Path(__file__).resolve().parent / "seeds"
TASKS = sorted(path.stem for path in SEEDS_DIR.glob("*.json"))

PROFILE = {
    "robot": "two robot arms (left, right). Degrees of freedom per arm: 6. Maximum gripper opening: about 9 cm. "
             "End effectors — left: parallel-jaw gripper; right: parallel-jaw gripper.",
    "frame": "Positions and moves use the task axes: +x forward away from the robot body, +y to the robot's own left, "
             "+z up, not image left/right. These axes stay fixed as the camera or gripper moves. "
             "Distances are in centimetres; angles are in degrees.",
    "gripper": "Positions refer to the grasp points: midway between the fingers, 3.5 cm short of their ends. "
               "'approach' is a unit vector pointing along the fingers; 'finger axis' is a unit vector along their "
               "opening/closing direction. Both use the task axes.",
}

# The robot frame in RoboTwin's world frame (+x right, +y away from the robot): rows are its axes.
FRAME = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
FRAME_ORIGIN_M = np.array([0.0, -0.65, 0.0])
GRASP_POINT_M = np.array([0.12, 0.0, 0.0])      # the grasp point in the wrist link frame: +x along the fingers
TABLE_SIZE_M = (1.2, 0.7)

# Pointing presets as wrist orientations in the world frame: fingers down, forward, and halfway between.
_HALF = float(np.sqrt(0.5))
_TOOL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])       # columns: finger axis, normal, approach
POINT = {name: FRAME.T @ np.array(axes) @ _TOOL.T for name, axes in {
    "down": [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
    "forward": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "down45": [[0.0, _HALF, _HALF], [1.0, 0.0, 0.0], [0.0, _HALF, -_HALF]],
}.items()}

OVERVIEW = {"width": 960, "height": 720, "fovy_deg": 60.0,
            "position": [0.0, -0.36, 1.50], "forward": [0.0, 0.42, -0.91], "left": [-1.0, 0.0, 0.0]}
WRIST_SIZE = (480, 360)
WRIST_FOVY_DEG = 63.0
# The wrist camera mount, once per arm in the aloha-agilex URDF (stock line, mounted line): centred above
# the gripper, pitched 25 degrees, looking down past the fingers.  The collision box and the inertia keep
# their place: their origins are shifted to compensate.
WRIST_MOUNT = (
    ('<origin xyz="0.07 0.032 0.065" rpy="0 0.4 0"/>', '<origin xyz="0.03 0 0.09" rpy="0 0.436332 0"/>'),
    ('<origin xyz="0 -0.032 0" rpy="1.57 0 1.57"/>', '<origin xyz="0 0 0" rpy="1.57 0 1.57"/>'),
    ('<origin xyz="0 -0.032 0" rpy="0 0 0"/>', '<origin xyz="0.04682 0 -0.00575" rpy="0 -0.03633 0"/>'),
)
WRIST_INERTIA = (r'(<mass value="0.1"/>\s*<origin) xyz="0 0 0"', r'\1 xyz="0.04682 0.032 -0.00575"')

CAPTIONS = {"overview": "overview: workspace camera.",
            "left_camera": "left_camera: camera on the left wrist or hand.",
            "right_camera": "right_camera: camera on the right wrist or hand."}
REACH_TOLERANCE = (1.5, 8.0)        # cm, degrees: a motion that ends farther from its target has failed
VIDEO_FPS = 5


@functools.lru_cache(maxsize=None)
def _embodiment_with_wrist_mount(robot_dir: str) -> str:
    """A copy of a RoboTwin embodiment directory whose URDF carries WRIST_MOUNT; RoboTwin itself is not modified."""
    source = Path(robot_dir).resolve()
    mirror = Path(tempfile.mkdtemp(prefix="nanoastra-embodiment-"))
    atexit.register(shutil.rmtree, mirror, ignore_errors=True)
    for entry in source.iterdir():
        if entry.name != "urdf":
            (mirror / entry.name).symlink_to(entry)
    (mirror / "urdf").mkdir()
    for entry in (source / "urdf").iterdir():
        if entry.suffix != ".urdf":
            (mirror / "urdf" / entry.name).symlink_to(entry)
            continue
        text = entry.read_text(encoding="utf-8")
        found = [text.count(stock) for stock, _ in WRIST_MOUNT]
        text, inertia = re.subn(*WRIST_INERTIA, text)
        if found + [inertia] != [2] * 4:
            raise ValueError(f"{entry}: not the aloha-agilex URDF of RoboTwin 2.0")
        for stock, mounted in WRIST_MOUNT:
            text = text.replace(stock, mounted)
        (mirror / "urdf" / entry.name).write_text(text, encoding="utf-8")
    return str(mirror)


def _task_args() -> dict:
    """The arguments RoboTwin's ``setup_demo`` takes, as ``script/eval_policy.py`` builds them."""
    def load(name):
        return yaml.safe_load((Path(CONFIGS_PATH) / name).read_text(encoding="utf-8"))

    args = load(f"{TASK_CONFIG}.yml")
    if args["embodiment"] != ["aloha-agilex"]:
        raise ValueError(f"unsupported embodiment {args['embodiment']!r}")
    args["camera"]["wrist_camera_type"] = "Large_D435"
    head = load("_camera_config.yml")[args["camera"]["head_camera_type"]]
    robot_dir = _embodiment_with_wrist_mount(load("_embodiment_config.yml")["aloha-agilex"]["file_path"])
    robot_config = yaml.safe_load((Path(robot_dir) / "config.yml").read_text(encoding="utf-8"))
    args.update(task_config=TASK_CONFIG, ckpt_setting="nanoastra", policy_name="nanoastra",
                head_camera_h=head["h"], head_camera_w=head["w"],
                left_robot_file=robot_dir, right_robot_file=robot_dir, dual_arm_embodied=True,
                left_embodiment_config=robot_config, right_embodiment_config=robot_config,
                eval_mode=True, render_freq=0, eval_video_save_dir=None, eval_video_log=False)
    return args


def _grasp_point(pose: np.ndarray) -> np.ndarray:
    return np.asarray(pose[:3], dtype=float) + t3d.quaternions.quat2mat(pose[3:]) @ GRASP_POINT_M


def _wrist_pose(grasp_point: np.ndarray, quat) -> np.ndarray:
    return np.concatenate([np.asarray(grasp_point, dtype=float) - t3d.quaternions.quat2mat(quat) @ GRASP_POINT_M, quat])


def _angle_deg(q1, q2) -> float:
    dot = abs(float(np.dot(np.asarray(q1) / np.linalg.norm(q1), np.asarray(q2) / np.linalg.norm(q2))))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def _world_axis(axes: tuple, axis: str, value: float) -> tuple[int, float]:
    """A signed amount along a robot-frame axis as (world axis index, signed amount)."""
    direction = FRAME[axes.index(axis)]
    index = int(np.argmax(np.abs(direction)))
    return index, float(direction[index]) * value


@dataclass
class Arm:
    """One arm in RoboTwin's world frame."""
    grasp_cm: np.ndarray
    wrist_cm: np.ndarray
    rotation: np.ndarray
    quat: np.ndarray
    commanded: float            # gripper opening, 0 closed .. 1 open
    measured: float


@dataclass
class View:
    name: str
    caption: str
    image: Image.Image


@dataclass
class Episode:
    index: int
    seed: int
    instruction: str


class RoboTwinEnv:
    profile = PROFILE

    def __init__(self, task: str, record_video: bool = True) -> None:
        self.task_name = task
        self.record_video = record_video
        self.args = {**_task_args(), "task_name": task}
        self.task = getattr(importlib.import_module(f"envs.{task}"), task)()
        cache = json.loads((SEEDS_DIR / f"{task}.json").read_text())
        self.seeds, self.infos = cache["seeds"], cache["infos"]
        self.frames: list[np.ndarray] = []
        self._open = False

    # ------------------------------------------------------------------ episodes
    def _expert(self, seed: int) -> tuple[bool, dict]:
        """Run RoboTwin's expert on ``seed``: (solved, the scene description it returns)."""
        try:
            self.task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **self.args)
            played = self.task.play_once()
            solved = bool(self.task.plan_success and self.task.check_success())
            self.task.close_env()
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, UnStableError):
                logger.warning("expert raised on seed %d: %s", seed, exc)
            try:
                self.task.close_env()
            except Exception:  # noqa: BLE001
                pass
            return False, {}
        return solved, played.get("info", {}) if isinstance(played, dict) else {}

    def reset(self, index: int) -> Episode:
        """Set up evaluation episode ``index``: the index-th seed of the task's list."""
        self.close_episode()
        if index >= len(self.seeds):
            raise IndexError(f"{self.task_name} has {len(self.seeds)} evaluation episodes")
        seed, info = int(self.seeds[index]), self.infos[index]
        # Some tasks set attributes in play_once() that check_success() reads, so the expert always runs first.
        solved, replayed = self._expert(seed)
        if not solved:
            logger.warning("the expert did not solve seed %d this time", seed)
        info = replayed or info

        self.task.setup_demo(now_ep_num=index, seed=seed, is_test=True, **self.args)
        self._open = True
        for camera in (self.task.cameras.left_camera, self.task.cameras.right_camera):
            camera.set_fovy(math.radians(WRIST_FOVY_DEG))
        self._add_overview_camera()
        self._count_physics_steps()
        # RoboTwin samples the wording from the global generator; seeding it gives each seed one instruction.
        state = random.getstate()
        random.seed(seed)
        try:
            candidates = generate_episode_descriptions(self.task_name, [info], 1)[0]["unseen"]
        finally:
            random.setstate(state)
        instruction = str(np.random.RandomState(seed).choice(candidates))
        self.task.set_instruction(instruction=instruction)
        self.task.get_obs()
        self.frames = []
        self._record_frame()
        return Episode(index, seed, instruction)

    def close_episode(self) -> None:
        if self._open:
            self.task.close_env(clear_cache=True)
            self._open = False

    def _add_overview_camera(self) -> None:
        import sapien

        forward = np.asarray(OVERVIEW["forward"], dtype=float)
        forward /= np.linalg.norm(forward)
        left = np.asarray(OVERVIEW["left"], dtype=float)
        left /= np.linalg.norm(left)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([forward, left, np.cross(forward, left)], axis=1)
        pose[:3, 3] = np.asarray(OVERVIEW["position"], dtype=float)
        pose[2, 3] += float(self.task.table_z_bias)
        self.overview = self.task.scene.add_camera(
            name="overview", width=OVERVIEW["width"], height=OVERVIEW["height"],
            fovy=math.radians(OVERVIEW["fovy_deg"]), near=0.1, far=100.0)
        self.overview.entity.set_pose(sapien.Pose(pose))

    def _overview_rgb(self) -> np.ndarray:
        self.overview.take_picture()
        rgba = self.overview.get_picture("Color")
        return np.ascontiguousarray((np.clip(rgba[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8))

    # ------------------------------------------------------------------ clock
    def _count_physics_steps(self) -> None:
        scene = self.task.scene
        self._steps = getattr(scene, "_nanoastra_steps", None)
        if self._steps is None:
            step, self._steps = scene.step, [0]

            def counted():
                self._steps[0] += 1
                return step()

            scene.step, scene._nanoastra_steps = counted, self._steps

    @property
    def physics_steps(self) -> int:
        return self._steps[0]

    @property
    def dt(self) -> float:
        return float(self.task.scene.get_timestep())

    def idle(self, steps: int) -> None:
        """Let up to ``steps`` physics steps pass with the robot holding still; ends early on success."""
        robot = self.task.robot
        hold = {"left": float(robot.get_left_gripper_val()), "right": float(robot.get_right_gripper_val())}
        for k in range(1, steps + 1):
            for arm, value in hold.items():
                robot.set_gripper(value, arm)
            self.task.scene.step()
            if k % 5 == 0 or k == steps:
                self._check_success()
            if self.success:
                return

    # ------------------------------------------------------------------ state
    @property
    def success(self) -> bool:
        return bool(self.task.eval_success)

    @property
    def steps_used(self) -> int:
        return int(self.task.take_action_cnt)

    @property
    def step_limit(self) -> int:
        return int(self.task.step_lim)

    @property
    def budget_exhausted(self) -> bool:
        return self.steps_used >= self.step_limit

    def _check_success(self) -> None:
        if not self.success:
            try:
                if self.task.check_success():
                    self.task.eval_success = True
            except Exception:  # noqa: BLE001
                pass

    def _measured_opening(self, arm: str) -> float:
        robot = self.task.robot
        entity = robot.left_entity if arm == "left" else robot.right_entity
        joint = (robot.left_gripper if arm == "left" else robot.right_gripper)[0][0]
        position = float(entity.get_qpos()[entity.get_active_joints().index(joint)])
        low, high = (float(value) for value in np.asarray(joint.get_limits()).reshape(-1)[:2])
        return float(np.clip((position - low) / (high - low), 0.0, 1.0))

    def _pose(self, arm: str) -> np.ndarray:
        return np.asarray(self.task.get_arm_pose(arm), dtype=float)

    def _commanded(self, arm: str) -> float:
        robot = self.task.robot
        return float(robot.get_left_gripper_val() if arm == "left" else robot.get_right_gripper_val())

    def _arm(self, arm: str) -> Arm:
        pose = self._pose(arm)
        rotation = t3d.quaternions.quat2mat(pose[3:])
        return Arm(grasp_cm=np.array([float(v) * 100.0 for v in pose[:3] + rotation @ GRASP_POINT_M]),
                   wrist_cm=np.array([float(v) * 100.0 for v in pose[:3]]), rotation=rotation, quat=pose[3:],
                   commanded=self._commanded(arm), measured=self._measured_opening(arm))

    def _arms(self) -> dict[str, Arm]:
        return {"left": self._arm("left"), "right": self._arm("right")}

    @staticmethod
    def _point(world_cm) -> list[float]:
        return (FRAME @ (np.asarray(world_cm) - FRAME_ORIGIN_M * 100)).tolist()

    def state(self) -> dict:
        """Both arms in the robot frame: grasp point (cm), approach and finger-axis unit vectors, gripper."""
        return {name: {"position_cm": self._point(arm.grasp_cm),
                       "approach": (FRAME @ arm.rotation[:, 0]).tolist(),
                       "finger_axis": (FRAME @ arm.rotation[:, 1]).tolist(),
                       "gripper_measured": arm.measured, "gripper_commanded": arm.commanded}
                for name, arm in self._arms().items()}

    @property
    def workspace(self) -> str:
        centre = self.task.table.get_pose().p
        low = [float(centre[i] - TABLE_SIZE_M[i] / 2) for i in (0, 1)]
        high = [float(centre[i] + TABLE_SIZE_M[i] / 2) for i in (0, 1)]
        x = (low[1] - FRAME_ORIGIN_M[1], high[1] - FRAME_ORIGIN_M[1])
        y = (0.0 - high[0], 0.0 - low[0])
        return (f"Table bounds: x from {x[0] * 100:g} to {x[1] * 100:g} cm, "
                f"y from {y[0] * 100:g} to {y[1] * 100:g} cm. Arm reach limits are unknown.")

    image_notes = overlay.NOTES.format(view="overview", plain="left_camera, right_camera")

    # ------------------------------------------------------------------ observation
    def observe(self) -> dict:
        """The state and the three images: the overview with coordinate guides, and the wrist cameras."""
        cameras = self.task.get_obs()["observation"]
        arms = self._arms()
        extrinsic = np.array(self.overview.get_extrinsic_matrix(), dtype=float, copy=True)
        extrinsic[:3, 3] += extrinsic[:3, :3] @ FRAME_ORIGIN_M
        extrinsic[:3, :3] = extrinsic[:3, :3] @ FRAME.T
        guides = {name: {"grasp": tuple(np.asarray(self._point(arm.grasp_cm), dtype=float) / 100),
                         "wrist": tuple(np.asarray(self._point(arm.wrist_cm), dtype=float) / 100)}
                  for name, arm in arms.items()}
        intrinsic = np.asarray(self.overview.get_intrinsic_matrix(), dtype=float)
        overview = overlay.enhance(Image.fromarray(self._overview_rgb()))
        views = [View("overview", CAPTIONS["overview"], overlay.draw_guides(overview, intrinsic, extrinsic, guides))]
        for name in ("left_camera", "right_camera"):
            image = Image.fromarray(np.ascontiguousarray(cameras[name]["rgb"]))
            if image.size != WRIST_SIZE:
                image = image.resize(WRIST_SIZE, Image.LANCZOS)
            views.append(View(name, CAPTIONS[name], overlay.enhance(image)))
        return {"state": self.state(), "views": views}

    # ------------------------------------------------------------------ actions
    def _target(self, arm: str, motions: list[Command]) -> np.ndarray:
        """The wrist pose ``arm`` reaches after ``motions``, each applied to the pose the previous one gives."""
        pose = self._pose(arm)
        for command in motions:
            point, quat = _grasp_point(pose), np.asarray(pose[3:], dtype=float).copy()
            if command.kind == "move":
                index, value = _world_axis(("x", "y", "z"), command.axis, command.value)
                point[index] += value / 100.0
            elif command.kind == "rotate":
                index, value = _world_axis(("roll", "pitch", "yaw"), command.axis, command.value)
                turn = t3d.axangles.axangle2mat(np.eye(3)[index].tolist(), math.radians(value))
                quat = t3d.quaternions.mat2quat(turn @ t3d.quaternions.quat2mat(quat))
            elif command.kind == "point":
                quat = t3d.quaternions.mat2quat(POINT[command.axis])
            else:       # home
                robot = self.task.robot
                home = np.asarray(robot.left_original_pose if arm == "left" else robot.right_original_pose, dtype=float)
                quat, point = home[3:].copy(), _grasp_point(home)
            pose = _wrist_pose(point, quat)
        return pose

    def _refusal(self) -> Optional[str]:
        if self.success:
            return "task already succeeded"
        if self.budget_exhausted:
            return "step budget exhausted"
        return None

    def _act(self, poses: dict, grippers: dict) -> Optional[str]:
        action = np.concatenate([poses["left"], [grippers["left"]], poses["right"], [grippers["right"]]])
        try:
            self.task.take_action(action.astype(np.float64), action_type="ee")
        except Exception as exc:  # noqa: BLE001
            logger.exception("take_action failed")
            return f"simulator error: {exc}"
        return None

    def wait(self) -> dict:
        """One step in which neither arm acts."""
        result = {"command": "wait", "kind": "wait", "ok": True, "note": ""}
        refusal = self._refusal()
        if refusal is None:
            refusal = self._act({arm: self._pose(arm) for arm in ("left", "right")},
                                {arm: self._commanded(arm) for arm in ("left", "right")})
            self._record_frame()
        if refusal is not None:
            result.update(ok=False, note=refusal)
        return result

    def execute(self, step: dict[str, ArmStep]) -> list[dict]:
        """One step: every arm in ``step`` moves to its target in one planned motion, the grippers moving
        meanwhile.  Returns one result per arm: ``ok``, and a ``note`` on what happened."""
        results = [{"command": part.text(), "kind": "motion" if part.motions else "gripper", "arm": arm, "ok": True,
                    "note": ""} for arm, part in step.items()]
        refusal = self._refusal()
        if refusal is not None:
            for result in results:
                result.update(ok=False, note=refusal)
            return results
        targets = {arm: self._target(arm, part.motions) for arm, part in step.items() if part.motions}
        poses = {arm: targets.get(arm, self._pose(arm)) for arm in ("left", "right")}
        grippers = {arm: self._commanded(arm) for arm in ("left", "right")}
        for arm, part in step.items():
            if arm in targets:
                poses[arm] = np.asarray(targets[arm], dtype=float)
            if part.gripper is not None:
                grippers[arm] = float(part.gripper)
        before = self._arms()
        error = self._act(poses, grippers)
        for arm, part in step.items():
            if part.gripper is not None and error is None and not self.success:
                self._settle_gripper(arm, float(part.gripper))
        after = self._arms()
        self._record_frame()
        for result in results:
            arm, part = result["arm"], step[result["arm"]]
            if error is not None:
                result.update(ok=False, note=error)
                continue
            if arm in targets:
                failure = self._missed(targets[arm], before[arm], after[arm])
                if failure:
                    result.update(ok=False, note=failure)
                    continue
            if part.gripper is not None:
                reading = gripper_reading(after[arm].measured, float(part.gripper))
                result["note"] = "gripper command executed; " + reading
        return results

    def _settle_gripper(self, arm: str, target: float, max_steps: int = 400, stable_steps: int = 20) -> None:
        """Hold every target and step physics until the fingers stop moving."""
        robot = self.task.robot
        other = "right" if arm == "left" else "left"
        other_target = self._commanded(other)
        previous, stable = self._measured_opening(arm), 0
        for _ in range(max_steps):
            robot.set_gripper(target, arm)
            robot.set_gripper(other_target, other)
            self.task.scene.step()
            current = self._measured_opening(arm)
            stable = stable + 1 if abs(current - previous) < 5e-4 else 0
            previous = current
            if stable >= stable_steps:
                break
        try:
            if self.task.check_success():
                self.task.eval_success = True
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _missed(target: np.ndarray, before: Arm, after: Arm) -> Optional[str]:
        """None when the grasp point ended within REACH_TOLERANCE of ``target``; otherwise what happened instead."""
        target = np.asarray(target, dtype=float)
        position_error = float(np.linalg.norm(after.grasp_cm - _grasp_point(target) * 100.0))
        angle_error = _angle_deg(after.quat, target[3:])
        if not (position_error > REACH_TOLERANCE[0] or angle_error > REACH_TOLERANCE[1]):
            return None
        moved = after.grasp_cm - before.grasp_cm
        distance = float(np.linalg.norm(moved))
        turned = _angle_deg(after.quat, before.quat)
        if distance < 0.5 and turned < 2.0:
            what = "the arm did NOT move"
        else:
            parts = []
            if angle_error > REACH_TOLERANCE[1]:
                parts.append(f"the gripper turned only {turned:.0f} deg")
            if distance >= 0.5:
                shown = FRAME @ np.array([float(f"{value:+.1f}") for value in moved])
                shown = ", ".join(f"{value + 0.0:+.1f}" for value in shown.tolist())
                parts.append(f"the grasp point moved ({shown}) cm"
                             + (", only part of the way" if position_error > REACH_TOLERANCE[0] else ""))
            what = " and ".join(parts) or f"the gripper turned only {turned:.0f} deg"
        return (f"motion planner could not reach the target: {what}; remaining error {position_error:.1f} cm / "
                f"{angle_error:.0f} deg (see CURRENT STATE for the actual pose). Try a smaller step, a different "
                "direction, or move away from the table/robot body first.")

    # ------------------------------------------------------------------ video
    def _record_frame(self) -> None:
        if self.record_video:
            self.task._update_render()
            self.frames.append(self._overview_rgb())

    def save_video(self, path: Path) -> None:
        import imageio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(path), self.frames, fps=VIDEO_FPS, codec="libx264", quality=6)
