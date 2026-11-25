"""The coordinate guides on a synthetic view: a camera 2 m up, looking straight down."""

import numpy as np
from PIL import Image

from nanoastra import overlay

INTRINSIC = [[800.0, 0.0, 480.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
EXTRINSIC = [[0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.5], [0.0, 0.0, -1.0, 2.0]]
ARMS = {"left": {"grasp": (0.5, 0.2, 0.9), "wrist": (0.38, 0.2, 0.9)},
        "right": {"grasp": (0.5, -0.2, 0.9), "wrist": (0.38, -0.2, 0.9)}}


def test_guides_mark_the_grasp_points():
    image = Image.new("RGB", (960, 720), (120, 120, 120))
    out = overlay.draw_guides(image, INTRINSIC, EXTRINSIC, ARMS)
    assert out.size == image.size and out.mode == "RGB"
    pixels = np.asarray(out).astype(int)
    # the left grasp point projects to (334.5, 360), the right one to (625.5, 360): a ring in the arm's colour
    left = pixels[348:372, 322:347].reshape(-1, 3)
    right = pixels[348:372, 613:638].reshape(-1, 3)
    assert ((left[:, 0] < 60) & (left[:, 1] > 150) & (left[:, 2] > 200)).any()
    assert ((right[:, 0] > 220) & (right[:, 1] > 90) & (right[:, 1] < 170) & (right[:, 2] < 60)).any()
    # far from the guides the image is untouched
    assert (pixels[700:, 440:520] == 120).all()


def test_an_arm_behind_the_camera_is_left_out():
    image = Image.new("RGB", (960, 720), (120, 120, 120))
    arms = {"left": {"grasp": (0.5, 0.2, 2.5), "wrist": (0.38, 0.2, 2.5)}}
    assert np.array_equal(np.asarray(overlay.draw_guides(image, INTRINSIC, EXTRINSIC, arms)), np.asarray(image))
