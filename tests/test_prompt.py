"""The prompt of a recorded evaluation episode is reproduced character for character."""

import json
from pathlib import Path

from nanoastra import overlay
from nanoastra.prompt import parse_reply, system_prompt, turn_text

EPISODE = json.loads((Path(__file__).parent / "data" / "episode.json").read_text(encoding="utf-8"))
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
CAPTIONS = ["overview: workspace camera.", "left_camera: camera on the left wrist or hand.",
            "right_camera: camera on the right wrist or hand."]


def test_system_prompt():
    notes = overlay.NOTES.format(view="overview", plain="left_camera, right_camera")
    workspace = "Table bounds: x from 30 to 100 cm, y from -60 to 60 cm. Arm reach limits are unknown."
    assert system_prompt(PROFILE, workspace, CAPTIONS, notes) == EPISODE["system_prompt"]


def test_turn_text():
    for turn in EPISODE["turns"]:
        text = turn_text(turn["turn"], EPISODE["instruction"], turn["state"], turn["last_results"],
                         ["overview", "left_camera", "right_camera"])
        assert text == turn["prompt"]


def test_parse_reply():
    reply = parse_reply(EPISODE["turns"][0]["reply"])
    assert reply["progress"] == "Grasp the red can with white top on the table."
    assert len(reply["steps"]) == 4
    assert parse_reply('Sure: {"progress": "a {b}", "steps": []} done') == {"progress": "a {b}", "steps": []}
