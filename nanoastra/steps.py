"""Semantic steps: the action part of a reply, and how it is carried out.

A reply holds an ordered list of steps.  A step gives each arm a list of commands that together
form one motion; the two arms act at the same time::

    "steps": [{"left": ["move x +5.0", "move z -2.0"], "right": ["wait"]},
              {"left": ["gripper close"], "right": ["wait"]}]

Commands::

    move x|y|z <cm>               rotate roll|pitch|yaw <deg>
    point down|forward|down45     gripper open|close|<0..1>
    home                          wait

The string ``"done"`` in place of a step requests a stop.  Parsing is lenient about case, spacing and units.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Optional

ARMS = ("left", "right")
STOP_NOTE = ("Stop requested; the controller continues this episode. Use the task and current observations "
             "to decide whether further action is appropriate.")
SKIPPED_NOTE = "not executed because an earlier one of this turn failed"
GRIPPER_WORDS = {"open": 1.0, "close": 0.0, "closed": 0.0, "half": 0.5}

_NUM = r"([-+]?\d+(?:\.\d+)?)"
_ARM = r"^(?:left|right)\s+"
_MOVE = re.compile(rf"{_ARM}move\s+([xyz])\s*{_NUM}\s*(cm|m)?$", re.I)
_ROTATE = re.compile(rf"{_ARM}rotate\s+(roll|pitch|yaw)\s*{_NUM}\s*(deg)?$", re.I)
_GRIPPER = re.compile(rf"{_ARM}gripper\s+(open|close|closed|half|{_NUM})$", re.I)
_POINT = re.compile(rf"{_ARM}point\s+(down|forward|down45)$", re.I)
_HOME = re.compile(rf"{_ARM}home$", re.I)


class CommandError(ValueError):
    """A command the parser refuses; the message is reported back to the model."""


@dataclass(frozen=True)
class Command:
    kind: str                       # move | rotate | point | home | gripper | wait
    axis: Optional[str] = None      # x|y|z, roll|pitch|yaw, or the name of a pointing preset
    value: Optional[float] = None   # cm, degrees, or gripper opening in [0, 1]

    def text(self) -> str:
        if self.kind in ("move", "rotate"):
            return f"{self.kind} {self.axis} {_signed(self.value)}"
        if self.kind == "point":
            return f"point {self.axis}"
        if self.kind == "gripper":
            return {0.0: "gripper close", 1.0: "gripper open"}.get(self.value, f"gripper {self.value:.2f}")
        return self.kind


def _signed(value: float) -> str:
    return f"{value:+.1f}" if round(value, 1) == round(value, 2) else f"{value:+.2f}"


def parse_command(arm: str, raw) -> Command:
    """One command string of ``arm``'s list."""
    if not isinstance(raw, str):
        raise CommandError(f"{arm} cell: a command is a string, got {raw!r}")
    text = raw.strip().strip("`").strip()
    first = text.split(" ", 1)[0].lower() if text else ""
    if first in ARMS:
        if first != arm:
            raise CommandError(f"{arm} cell holds a {first}-arm command {raw!r}; put it in the {first} column")
        text = text.split(" ", 1)[1] if " " in text else ""
    if text.lower() in ("wait", ""):
        return Command("wait")
    if text.lower() == "done":
        raise CommandError(f'{arm} cell: "done" is a step of its own, not a cell command')
    line = f"{arm} {text}"
    text = re.sub(r"\s+", " ", line.strip().strip("`").strip())
    text = text.split("#", 1)[0].strip().rstrip(".").strip()
    if match := _MOVE.match(text):
        value = float(match[2]) * (100.0 if (match[3] or "cm").lower() == "m" else 1.0)
        if not math.isfinite(value):
            raise CommandError("move magnitude must be finite")
        return Command("move", match[1].lower(), value)
    if match := _ROTATE.match(text):
        value = float(match[2])
        if not math.isfinite(value):
            raise CommandError("rotation magnitude must be finite")
        return Command("rotate", match[1].lower(), value)
    if match := _GRIPPER.match(text):
        word = match[1].lower()
        value = GRIPPER_WORDS[word] if word in GRIPPER_WORDS else float(word)
        if not math.isfinite(value):
            raise CommandError("gripper opening must be finite")
        return Command("gripper", value=max(0.0, min(1.0, value)))
    if match := _POINT.match(text):
        return Command("point", match[1].lower())
    if _HOME.match(text):
        return Command("home")
    if re.search(r"\s\+\s", text):
        raise CommandError(f"{line!r} runs several commands together; send ONE command per string")
    raise CommandError(f"unrecognised command: {line!r}")


@dataclass
class ArmStep:
    """What one arm does in one step: one motion, with the gripper moving meanwhile."""
    arm: str
    motions: list[Command]              # rotate / point / home in the order written, then one move per axis
    gripper: Optional[float] = None

    def text(self) -> str:
        parts = [command.text() for command in self.motions]
        if self.gripper is not None:
            parts.append(Command("gripper", value=self.gripper).text())
        return f"{self.arm} [" + ", ".join(json.dumps(part) for part in parts) + "]"


def parse_arm(arm: str, raw) -> Optional[ArmStep]:
    """One arm's command list of a step; None when the arm waits."""
    if raw is None:
        return None
    moves = {"x": 0.0, "y": 0.0, "z": 0.0}
    posed: list[Command] = []
    gripper: Optional[float] = None
    for item in raw if isinstance(raw, list) else [raw]:
        command = parse_command(arm, item)
        if command.kind == "gripper":
            if gripper is not None:
                raise CommandError(f"{arm} cell has two gripper commands")
            gripper = command.value
        elif command.kind == "move":
            moves[command.axis] += command.value
        elif command.kind != "wait":
            posed.append(command)
    merged = []
    for axis, total in moves.items():
        if total:
            if not math.isfinite(total):
                raise CommandError(f"{arm} cell: total move on {axis} must be finite")
            merged.append(Command("move", axis, total))
    if posed or merged or gripper is not None:
        return ArmStep(arm, posed + merged, gripper)
    return None


@dataclass
class Step:
    left: Optional[ArmStep]
    right: Optional[ArmStep]

    @property
    def arms(self) -> list[ArmStep]:
        return [arm for arm in (self.left, self.right) if arm is not None]


@dataclass
class Invalid:
    """A step the parser refused.  It keeps its place in the list and is reported there."""
    index: int
    raw: object
    error: str

    def result(self) -> dict:
        command = self.raw.strip() if isinstance(self.raw, str) else f"step {self.index}"
        return {"command": command, "kind": "invalid", "ok": False, "note": self.error}


DONE = "done"


def parse_steps(reply: dict) -> list:
    """The "steps" of a parsed reply as a list of Step, Invalid and DONE."""
    raw = reply.get("steps")
    if raw is None:
        raw = []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return [Invalid(1, raw, f"'steps' must be a list of rows, got {type(raw).__name__}")]
    steps = []
    for index, row in enumerate(raw, 1):
        if isinstance(row, str) and row.strip().lower().strip('"') == DONE:
            steps.append(DONE)
            continue
        if isinstance(row, str):
            error = f'a step is an object {{"left": ..., "right": ...}}, not the text {row!r}'
        elif not isinstance(row, dict):
            error = f"not an object ({row!r})"
        elif any(str(key).lower() not in ARMS for key in row):
            unknown = [key for key in row if str(key).lower() not in ARMS]
            error = f'unknown keys {unknown}; a row has exactly "left" and "right"'
        else:
            row = {str(key).lower(): value for key, value in row.items()}
            try:
                steps.append(Step(parse_arm("left", row.get("left")), parse_arm("right", row.get("right"))))
                continue
            except CommandError as exc:
                error = str(exc)
        steps.append(Invalid(index, row, error))
    return steps


def _skipped(command: str) -> dict:
    return {"command": command, "kind": "skipped", "ok": False, "note": SKIPPED_NOTE}


def execute_steps(env, steps: list, gate: Optional[Callable[[int], bool]] = None) -> tuple[list[dict], bool]:
    """Carry out a turn's steps in order.  Returns (results, stop requested).

    A failed or invalid step ends the turn: the steps after it are reported as skipped.
    ``gate(i)`` is called before step ``i`` (1-based) and may wait for it; True ends the turn there.
    """
    results: list[dict] = []
    failed = False
    for index, step in enumerate(steps, 1):
        if isinstance(step, Invalid):
            results.append(step.result())
            failed = True
            continue
        if gate is not None and not failed and gate(index):
            return results, False
        if step == DONE:
            if failed:
                results.append(_skipped(DONE))
                continue
            results.append({"command": DONE, "kind": "done", "ok": False, "note": STOP_NOTE})
            return results, True
        if not step.arms:
            name = f"step {index}: both wait"
            results.append(_skipped(name) if failed else {**env.wait(), "command": name})
            continue
        if failed:
            results += [_skipped(f"step {index}: {arm.text()}") for arm in step.arms]
            continue
        done = env.execute({arm.arm: arm for arm in step.arms})
        for result in done:
            result["command"] = f"step {index}: {result['command']}"
        results += done
        if env.success or env.budget_exhausted:
            return results, False
        failed = any(not result["ok"] for result in done)
    return results, False
