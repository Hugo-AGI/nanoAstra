"""Blocking and streaming execution on a scripted robot and a scripted model."""

import json
from types import SimpleNamespace

from PIL import Image

from nanoastra.agent import Agent
from nanoastra.client import Reply
from nanoastra.streaming import CONTINUE, OPENING, step_spans

STEP_PHYSICS = 250                  # every step takes one second of robot time
HEAD = '```json\n{\n "progress": "Move the cup.",\n "steps"'
TAIL = "\n ]\n}\n```"


def step(k: int) -> str:
    return json.dumps({"left": [f"move x +{k}.0"], "right": ["wait"]})


class Robot:
    """Succeeds once six steps have run."""

    profile = {"robot": "r", "frame": "f", "gripper": "g"}
    workspace, image_notes, dt, step_limit = "w", "", 0.004, 400

    def __init__(self):
        self.physics_steps, self.executed = 0, []

    success = property(lambda self: len(self.executed) >= 6)
    steps_used = property(lambda self: len(self.executed))
    budget_exhausted = False

    def observe(self):
        arm = {"position_cm": [0, 0, 0], "approach": [1, 0, 0], "finger_axis": [0, 1, 0],
               "gripper_measured": 1.0, "gripper_commanded": 1.0}
        return {"state": {"left": arm, "right": arm},
                "views": [SimpleNamespace(name="overview", caption="overview: c.", image=Image.new("RGB", (8, 8)))]}

    def idle(self, steps):
        self.physics_steps += steps

    def execute(self, parts):
        self.physics_steps += STEP_PHYSICS
        self.executed.append((parts["left"].text(), round(self.physics_steps * self.dt, 3)))
        return [{"command": parts["left"].text(), "kind": "motion", "arm": "left", "ok": True, "note": ""}]


class Model:
    """Writes four new steps per reply (two when it continues a prefilled one), numbered in the order they are
    written; one step per 20 tokens."""

    def __init__(self):
        self.requests, self.written = [], 0

    def _steps(self, n):
        self.written += n
        return [step(k) for k in range(self.written - n + 1, self.written + 1)]

    def chat(self, messages, extra=None):
        self.requests.append((messages, extra))
        if extra == OPENING:
            return Reply(text=HEAD, prompt_tokens=3000, completion_tokens=15)
        if extra == CONTINUE:
            text = '"]}' + "".join(",\n  " + s for s in self._steps(2)) + TAIL
        else:
            text = HEAD + ": [\n  " + ",\n  ".join(self._steps(4)) + TAIL
        ends = [end for _, end in step_spans(text if extra != CONTINUE else HEAD + ': [\n  {"left": ["x' + text)]
        offset = 0 if extra != CONTINUE else len(HEAD + ': [\n  {"left": ["x')
        chunks = [(end - offset, 20 * (i + 1)) for i, end in enumerate(ends) if end > offset]
        chunks.append((len(text), chunks[-1][1] + 10))
        return Reply(text=text, prompt_tokens=3000, completion_tokens=chunks[-1][1], chunks=chunks)


def run(mode):
    robot, model = Robot(), Model()
    result = Agent(model, mode).run_episode(robot, "Move the cup.")
    return robot, model, result


def test_blocking_waits_for_the_whole_reply():
    robot, model, result = run("blocking")
    assert result.success and result.turns == 2 and len(model.requests) == 2
    # 3000 prompt tokens and 90 output tokens take 1.003 s; the first step ends one second later
    assert robot.executed[0] == ('left ["move x +1.0"]', 2.004)


def test_streaming_starts_each_step_when_it_is_written():
    robot, model, result = run("streaming")
    assert result.success
    assert [text for text, _ in robot.executed] == [f'left ["move x +{k}.0"]' for k in range(1, 7)]
    # the first step is complete after 20 tokens (0.552 s)
    assert robot.executed[0][1] == 1.552
    blocking = run("blocking")[2]
    assert result.success_seconds < blocking.success_seconds


def test_streaming_prefills_the_steps_still_to_run():
    robot, model, result = run("streaming")
    (first, _), (_, opening), (continued, extra) = model.requests[:3]
    assert opening == OPENING and extra == CONTINUE
    # the second observation was sent with two steps left; the model continues its own opening followed by them
    prefix = continued[-1]
    assert prefix["role"] == "assistant" and prefix["content"].startswith(HEAD)
    assert prefix["content"].endswith(step(4)[:-3])
    # and the conversation shows the first turn as having sent only the two steps that had run
    shown = continued[2]
    assert shown["role"] == "assistant" and step(2) in shown["content"] and step(3) not in shown["content"]
    kept = json.loads(shown["content"].strip("`").removeprefix("json"))["steps"]
    assert kept == [json.loads(step(1)), json.loads(step(2))]
