"""The text the model reads and the reply it writes.

The system prompt describes the robot, its coordinate frame, the images and the action vocabulary;
each turn's user message carries the task, the outcome of the previous steps and the robot state.
The reply is one JSON object: the current progress and the next steps.
"""

from __future__ import annotations

import json
import math

GRIPPER_GUIDANCE = ("Opening ranges from 0 (closed) to 1 (open). 'Measured opening' is sensor feedback; "
                    "'commanded opening' is the requested setting. "
                    "Use the images to check the grasp, not opening alone.")

ACTIONS = (
    'ACTIONS: "steps" is an ordered list. Each step has a command list for each arm:\n'
    '{"left": [...], "right": [...]}. The two arms act simultaneously.\n'
    "Each arm's list defines one combined motion:\n"
    "- rotate, point and home set the pose first, in the order written.\n"
    '- Moves then add up along each axis from that pose. For example, ["move z +5", "move z -14"] gives one move '
    "of -9 cm, not two separate moves.\n"
    "- The motion planner chooses the path to the target; it may not be straight.\n"
    '- The gripper opens or closes during the motion. To close after arriving, put "gripper close" in the next step.\n'
    'Use ["wait"] for an inactive arm.\n'
    "\n"
    "Commands (one string each; omit the arm name):\n"
    "  move x|y|z <cm>              move along a task axis without changing orientation.\n"
    "  rotate roll|pitch|yaw <deg>  rotate about task x|y|z respectively, through its reference point.\n"
    "  point down|forward|down45   point the fingers down (-z), forward (+x), or halfway between; "
    "set the finger axis to +y.\n"
    "  gripper open|close|<0..1>   set the opening: 0 is closed, 1 is open.\n"
    "  home                       return to the arm's initial position and orientation.\n"
    "  wait                       take no action.\n"
    'Example: [{"left": ["move x +5"], "right": ["move x +5"]},\n'
    '          {"left": ["gripper close"], "right": ["gripper close"]}]\n'
    'To request a stop, include the string "done" in "steps".\n'
)

CONVERSATION = ("CONVERSATION: earlier messages, when present, show the previous turn (without images). "
                "Use it to track changes and past attempts. Choose the next actions from the current observation.")

RESPONSE_FORMAT = (
    "RESPONSE FORMAT: reply with ONE JSON object and nothing else:\n"
    "{\n"
    '  "progress": "<the current subtask, without a prefix>",\n'
    '  "steps": [{"left": ["<command>", ...], "right": ["<command>", ...]}, ...]\n'
    "}\n"
    "If a step is FAILED or INVALID, the remaining steps in this turn are SKIPPED. "
    "Check the next observation before continuing.\n"
    'Use "done" to request a stop when no further action is needed. It does not confirm success. '
    "If another observation follows, reassess the task before continuing."
)


def system_prompt(profile: dict, workspace: str, captions: list[str], image_notes: str = "") -> str:
    """``profile``: the robot's ``robot``, ``frame`` and ``gripper`` descriptions; ``captions``: one line per
    image, in the order they are attached; ``image_notes``: what is drawn on them."""
    images = "IMAGES (attached every turn, in this order):\n" + "\n".join(f"- {caption}" for caption in captions)
    if image_notes:
        images += "\n" + image_notes
    return "\n\n".join((
        "Use the images and state to complete the task. Reply with the next steps to execute.",
        f"ROBOT: {profile['robot']}",
        f"COORDINATE FRAME: {profile['frame']}",
        f"WORKSPACE: {workspace}",
        images,
        f"GRIPPER: {profile['gripper']}\n{GRIPPER_GUIDANCE}",
        f"{ACTIONS}\n{CONVERSATION}",
        RESPONSE_FORMAT,
    ))


def _vector(values, digits: int) -> str:
    return str(tuple(round(float(value), digits) + 0.0 for value in values))


def _opening(value) -> str:
    if value is None or not math.isfinite(float(value)):
        return "unavailable"
    return f"{min(1.0, max(0.0, float(value))):.2f}"


def gripper_reading(measured, commanded) -> str:
    return f"measured opening = {_opening(measured)}, commanded opening = {_opening(commanded)}"


def state_text(state: dict) -> str:
    """One line per arm: grasp point (cm), approach and finger axis (unit vectors), gripper opening."""
    return "\n".join(
        f"{arm.upper()} grasp point = {_vector(state[arm]['position_cm'], 1)} cm, "
        f"approach = {_vector(state[arm]['approach'], 2)}, finger axis = {_vector(state[arm]['finger_axis'], 2)}, "
        + gripper_reading(state[arm]["gripper_measured"], state[arm]["gripper_commanded"])
        for arm in ("left", "right") if arm in state)


def _outcome(result: dict) -> str:
    kind = result.get("kind")
    if kind == "empty":
        return "nothing was executed"
    if kind in ("skipped", "invalid"):
        return kind.upper()
    return "ok" if result.get("ok") else "FAILED"


def turn_core(turn: int, instruction: str, state: dict, last_results: list[dict]) -> str:
    """The part of a turn's message that stays in the conversation: task, turn, last results, state."""
    parts = [f"TASK: {instruction}", f"TURN {turn}."]
    if last_results:
        lines = [f"- {result['command']}: {_outcome(result)}" + (f" ({result['note']})" if result.get("note") else "")
                 for result in last_results]
        parts.append("RESULT OF YOUR LAST TURN:\n" + "\n".join(lines))
    parts.append("CURRENT STATE:\n" + state_text(state))
    return "\n\n".join(parts)


def turn_text(turn: int, instruction: str, state: dict, last_results: list[dict], image_names: list[str]) -> str:
    return "\n\n".join((turn_core(turn, instruction, state, last_results),
                        "IMAGES: " + ", ".join(image_names) + ".",
                        "Reply with the JSON object."))


def parse_reply(text: str) -> dict:
    """The first JSON object in a reply (code fences and surrounding prose are tolerated)."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:index + 1])
    raise ValueError("unbalanced JSON object in reply")
