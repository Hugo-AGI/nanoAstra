"""Parsing and executing steps."""

from nanoastra.steps import DONE, Invalid, Step, execute_steps, parse_steps


class Robot:
    """Records what it is asked to do; a step that moves the arm named in ``failing`` fails."""

    success = budget_exhausted = False

    def __init__(self, failing=None):
        self.failing, self.done = failing, []

    def wait(self):
        self.done.append("wait")
        return {"command": "wait", "kind": "wait", "ok": True, "note": ""}

    def execute(self, step):
        self.done.append({arm: part.text() for arm, part in step.items()})
        return [{"command": part.text(), "kind": "motion", "arm": arm, "ok": arm != self.failing, "note": ""}
                for arm, part in step.items()]


def test_moves_add_up_and_poses_come_first():
    step, = parse_steps({"steps": [{"left": ["move z +5", "gripper close", "point down", "move z -14", "move x 2cm"],
                                    "right": ["wait"]}]})
    assert isinstance(step, Step) and step.right is None
    assert step.left.text() == 'left ["point down", "move x +2.0", "move z -9.0", "gripper close"]'


def test_refused_steps_keep_their_place():
    steps = parse_steps({"steps": [{"left": ["move x 1"], "right": ["wait"]},
                                   {"left": ["jump"], "right": ["wait"]},
                                   {"left": ["right move x 1"], "right": ["wait"]},
                                   {"left": ["move x 1"], "both": ["wait"]},
                                   "done"]})
    assert [type(step).__name__ for step in steps[:4]] == ["Step", "Invalid", "Invalid", "Invalid"]
    assert steps[4] == DONE
    assert steps[1].error == "unrecognised command: 'left jump'"
    assert "put it in the right column" in steps[2].error


def test_a_failure_ends_the_turn():
    steps = parse_steps({"steps": [{"left": ["move x 1"], "right": ["move y 1"]},
                                   {"left": ["wait"], "right": ["wait"]},
                                   {"left": ["gripper open"], "right": ["wait"]},
                                   "done"]})
    robot = Robot(failing="right")
    results, stop = execute_steps(robot, steps)
    assert not stop and len(robot.done) == 1
    assert [(result["command"], result["kind"], result["ok"]) for result in results] == [
        ('step 1: left ["move x +1.0"]', "motion", True),
        ('step 1: right ["move y +1.0"]', "motion", False),
        ("step 2: both wait", "skipped", False),
        ('step 3: left ["gripper open"]', "skipped", False),
        ("done", "skipped", False),
    ]


def test_done_requests_a_stop():
    robot = Robot()
    results, stop = execute_steps(robot, parse_steps({"steps": [{"left": ["wait"], "right": ["wait"]}, "done"]}))
    assert stop and robot.done == ["wait"] and results[-1]["kind"] == "done"


def test_steps_must_be_a_list():
    step, = parse_steps({"steps": 3})
    assert isinstance(step, Invalid) and step.result()["command"] == "step 1"
