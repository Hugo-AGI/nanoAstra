"""Coordinate guides drawn on the overview image.

For each arm: a ring at the grasp point, dashed lines through it along x and y at its own height and a
line going down along z, with ticks every 5 cm and coordinate labels every 10 cm.  A legend in one corner
shows the image directions of +x, +y and +z.

Everything is drawn on a supersampled transparent layer laid out for a 640x480 view and composited
onto the image at its own resolution.
"""

from __future__ import annotations

import math
from itertools import groupby

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ARM = {"left": (0, 186, 240), "right": (255, 128, 0)}
ARM_TEXT = {"left": (150, 228, 255), "right": (255, 196, 130)}
AXIS = {"x": (196, 44, 44), "y": (28, 128, 62), "z": (37, 99, 235)}
INK = (18, 24, 22)
WHITE = (255, 255, 255, 255)
LABEL_FILL = (*INK, 76)
SUPERSAMPLE = 3
REACH_CM, DROP_CM, TICK_CM = 20, 25, 5

NOTES = (
    "Drawn on them:\n"
    "- in {view}, cyan guides belong to the left arm and orange to the right. L/R circles mark each grasp point. "
    "Dashed x/y lines cross at each grasp point's height; the z line runs downward. "
    f"Ticks are {TICK_CM} cm apart, with coordinate labels every {2 * TICK_CM} cm. "
    "Compare positions at the same height: objects above or below the guides appear shifted by perspective;\n"
    "- the corner legend shows the image directions of +x, +y and +z; an end-on axis is a dot;\n"
    "- follow short leader lines from offset labels to their ticks; not every tick is labelled.\n"
    "No coordinate guides are drawn in: {plain}."
)


def enhance(image: Image.Image) -> Image.Image:
    """The contrast stretch applied to every image before it is annotated and sent."""
    return ImageOps.autocontrast(image.convert("RGB"), cutoff=1, preserve_tone=True)


def draw_guides(image: Image.Image, intrinsic, extrinsic, arms: dict) -> Image.Image:
    """``image`` with the guides of ``arms`` = {name: {"grasp": xyz, "wrist": xyz}} (metres, in the frame
    of ``extrinsic``, the 3x4 camera-from-frame matrix; ``intrinsic``: the 3x3 pinhole matrix of ``image``)."""
    ratio = 480 / min(image.size)
    size = (max(1, round(image.width * ratio)), max(1, round(image.height * ratio)))
    sx, sy = size[0] / image.width, size[1] / image.height
    K = np.array([[intrinsic[0][0] * sx, 0, intrinsic[0][2] * sx], [0, intrinsic[1][1] * sy, intrinsic[1][2] * sy],
                  [0, 0, 1.0]])
    E = np.asarray(extrinsic, dtype=np.float64)[:3]
    canvas = _Canvas(size, image.resize(size, Image.Resampling.LANCZOS))
    labels = _Labels(canvas)
    px, grasps = 13, []
    for arm in arms.values():
        labels.protect(_project(K, E, arm["grasp"]), 8)
    for name, arm in arms.items():
        color, tint = ARM[name], ARM_TEXT[name]
        g = np.asarray(arm["grasp"], dtype=float) * 100
        tip = _project(K, E, g / 100)
        if tip is None:
            continue
        grasps.append(g / 100)
        side = -1.0 if name == "left" else 1.0
        for axis, index, across in (("x", 0, np.array([0, side, 0])), ("y", 1, np.array([side, 0, 0]))):
            lo = 10 * math.floor((g[index] - REACH_CM) / 10)
            hi = 10 * math.ceil((g[index] + REACH_CM) / 10)
            _guide(canvas, labels, K, E, g, axis, lo, hi, across, color, tint, px)
        _guide(canvas, labels, K, E, g, "z", g[2] - DROP_CM, g[2], np.array([-side, 0, 0]), color, tint, px,
               priority=0.5)
        wrist = _project(K, E, arm["wrist"])
        if wrist is not None:
            canvas.line([wrist, tip], color, 2.0, halo=0.8)
        canvas.ring(tip, 5.0, color, 2.0)
        labels.put([(tip[0] + 22, tip[1] - 16), (tip[0] - 22, tip[1] - 16)], name[0].upper(), px,
                   fill=(*color, 255), text_fill=INK if name == "left" else WHITE)
    if grasps:
        _legend(canvas, labels, K, E, np.mean(grasps, axis=0), px)
    labels.flush()
    return canvas.composite(image)


def _project(K, E, point):
    """Pixel of a point (metres); None behind the camera."""
    c = E[:, :3] @ np.asarray(point, dtype=np.float64) + E[:, 3]
    if not np.all(np.isfinite(c)) or c[2] <= 1e-6:
        return None
    uv = K @ (c / c[2])
    return float(uv[0]), float(uv[1])


def _image_direction(K, E, point, direction):
    a, b = _project(K, E, point), _project(K, E, np.asarray(point) + np.asarray(direction) * 0.01)
    if a is None or b is None:
        return None
    d = np.subtract(b, a)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else None


class _Canvas:
    """A supersampled RGBA layer; coordinates are given in layout pixels."""

    def __init__(self, size, image) -> None:
        self.W, self.H = size
        self.layer = Image.new("RGBA", (self.W * SUPERSAMPLE, self.H * SUPERSAMPLE), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)
        self._fonts: dict = {}
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        edges = np.zeros_like(gray)
        edges[:, 1:] += np.abs(np.diff(gray, axis=1))
        edges[1:, :] += np.abs(np.diff(gray, axis=0))
        self.edges = np.pad(np.minimum(edges, 64).cumsum(0, dtype=np.float64).cumsum(1), ((1, 0), (1, 0)))

    def clutter(self, box) -> float:
        """Mean edge strength of the image under ``box``: labels prefer plain regions."""
        left, top, right, bottom = (int(round(value)) for value in box)
        left, right = max(0, left), min(self.W, right)
        top, bottom = max(0, top), min(self.H, bottom)
        if left >= right or top >= bottom:
            return float("inf")
        total = self.edges[bottom, right] - self.edges[top, right] - self.edges[bottom, left] + self.edges[top, left]
        return float(total / ((right - left) * (bottom - top)))

    def font(self, px: int):
        if px not in self._fonts:
            self._fonts[px] = ImageFont.load_default(size=px * SUPERSAMPLE)
        return self._fonts[px]

    def line(self, points, color, width=2.0, halo=1.0, dash=None) -> None:
        for visible, group in groupby(points, key=lambda p: p is not None):
            strip = list(group)
            if not visible or len(strip) < 2:
                continue
            for segment in ([strip] if not dash else _dashed(strip, dash)):
                scaled = [(p[0] * SUPERSAMPLE, p[1] * SUPERSAMPLE) for p in segment]
                if halo:
                    self.d.line(scaled, fill=(0, 0, 0, 190), width=int((width + 2 * halo) * SUPERSAMPLE), joint="curve")
                self.d.line(scaled, fill=(*color, 255), width=max(1, int(width * SUPERSAMPLE)), joint="curve")

    def dot(self, p, r, fill, ring=None, ring_width=1.5) -> None:
        x, y = p[0] * SUPERSAMPLE, p[1] * SUPERSAMPLE
        if ring:
            q = (r + ring_width) * SUPERSAMPLE
            self.d.ellipse([x - q, y - q, x + q, y + q], fill=(*ring, 255))
        q = r * SUPERSAMPLE
        self.d.ellipse([x - q, y - q, x + q, y + q], fill=(*fill, 255))

    def ring(self, p, r, color, width=1.5) -> None:
        x, y, q = p[0] * SUPERSAMPLE, p[1] * SUPERSAMPLE, r * SUPERSAMPLE
        self.d.ellipse([x - q, y - q, x + q, y + q], outline=(0, 0, 0, 178), width=int((width + 2) * SUPERSAMPLE))
        self.d.ellipse([x - q, y - q, x + q, y + q], outline=(*color, 255), width=int(width * SUPERSAMPLE))

    def arrow(self, origin, tip, color) -> None:
        u = np.subtract(tip, origin)
        u = u / (np.linalg.norm(u) + 1e-9)
        n = np.array([-u[1], u[0]])
        origin, tip = np.asarray(origin), np.asarray(tip)
        base = tip - u * 9
        outline = [origin + n * 1.5, base + n * 1.5, base + n * 5, tip, base - n * 5, base - n * 1.5, origin - n * 1.5]
        scaled = [(p[0] * SUPERSAMPLE, p[1] * SUPERSAMPLE) for p in outline]
        self.d.line(scaled + [scaled[0]], fill=(0, 0, 0, 190), width=max(1, round(2.4 * SUPERSAMPLE)), joint="curve")
        self.d.polygon(scaled, fill=(*color, 255))

    def text_box(self, center, text, px):
        left, top, right, bottom = self.d.textbbox(
            (center[0] * SUPERSAMPLE, center[1] * SUPERSAMPLE), text, font=self.font(px), anchor="mm",
            stroke_width=SUPERSAMPLE)
        return (left / SUPERSAMPLE - 2, top / SUPERSAMPLE - 1, right / SUPERSAMPLE + 2, bottom / SUPERSAMPLE + 1)

    def label(self, center, text, px, fill=None, text_fill=WHITE):
        box = self.text_box(center, text, px)
        self.d.rounded_rectangle([v * SUPERSAMPLE for v in box], radius=2 * SUPERSAMPLE, fill=fill or LABEL_FILL)
        self.d.text((center[0] * SUPERSAMPLE, center[1] * SUPERSAMPLE), text, font=self.font(px),
                    fill=(*text_fill[:3], 255), anchor="mm", stroke_width=0 if fill else SUPERSAMPLE,
                    stroke_fill=(*INK, 255))
        return box

    def composite(self, image):
        out = image.convert("RGBA")
        out.alpha_composite(self.layer.resize(image.size, Image.Resampling.LANCZOS))
        return out.convert("RGB")


def _dashed(points, dash):
    on, off = dash
    out, current = [], []
    used, drawing = 0.0, True
    for a, b in zip(points, points[1:]):
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        if length < 1e-6:
            continue
        position = 0.0
        while position < length:
            span = min(length - position, (on if drawing else off) - used)
            p0 = (a[0] + (b[0] - a[0]) * position / length, a[1] + (b[1] - a[1]) * position / length)
            p1 = (a[0] + (b[0] - a[0]) * (position + span) / length, a[1] + (b[1] - a[1]) * (position + span) / length)
            if drawing:
                current = current or [p0]
                current.append(p1)
            position += span
            used += span
            if used >= (on if drawing else off) - 1e-9:
                used, drawing = 0.0, not drawing
                if current and not drawing:
                    out.append(current)
                    current = []
    if current:
        out.append(current)
    return out


def _crosses(a, b, box, margin=2.0) -> bool:
    """Whether the segment a-b passes through ``box``."""
    lo, hi = 0.0, 1.0
    for axis in (0, 1):
        lower, upper = box[axis] - margin, box[axis + 2] + margin
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-9:
            if not lower <= a[axis] <= upper:
                return False
        else:
            enter, leave = sorted(((lower - a[axis]) / delta, (upper - a[axis]) / delta))
            lo, hi = max(lo, enter), min(hi, leave)
            if lo > hi:
                return False
    return True


class _Labels:
    """Labels that do not overlap: queued while the lines are drawn, then placed by priority; one that
    does not fit anywhere is left out."""

    def __init__(self, canvas: _Canvas) -> None:
        self.canvas = canvas
        self.boxes: list[tuple] = []
        self.queue: list[tuple] = []
        self.leaders: list[tuple] = []
        self.protected: list[tuple] = []

    def protect(self, point, radius: float) -> None:
        if point is not None:
            self.protected.append((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius))

    def fits(self, box) -> bool:
        if box[0] < 2 or box[1] < 2 or box[2] > self.canvas.W - 2 or box[3] > self.canvas.H - 2:
            return False
        return not any(not (box[2] + 3 < o[0] or o[2] + 3 < box[0] or box[3] + 3 < o[1] or o[3] + 3 < box[1])
                       for o in [*self.boxes, *self.protected]) and not any(
                           _crosses(a, b, box) for a, b in self.leaders)

    def put(self, centers, text, px, fill=None, text_fill=WHITE, anchor=None, leader_color=None) -> bool:
        """Place the label at the first of ``centers`` that fits, least cluttered first; a label away from
        its preferred place gets a short leader line to ``anchor``."""
        choices = list(enumerate([centers] if isinstance(centers, tuple) else centers))
        choices.sort(key=lambda entry: self.canvas.clutter(self.canvas.text_box(entry[1], text, px)) + entry[0] * 1.5)
        for index, center in choices:
            box = self.canvas.text_box(center, text, px)
            if not self.fits(box):
                continue
            if anchor is not None and index > 0:
                end = tuple(float(np.clip(anchor[i], box[i], box[i + 2])) for i in (0, 1))
                if any(_crosses(anchor, end, prior) for prior in self.boxes):
                    continue
                if any(_crosses(anchor, end, (*np.minimum(a, b), *np.maximum(a, b))) for a, b in self.leaders):
                    continue
                self.canvas.line([anchor, end], leader_color, 0.8, halo=0.4)
                self.leaders.append((anchor, end))
            self.boxes.append(self.canvas.label(center, text, px, fill, text_fill))
            return True
        return False

    def defer(self, priority, centers, text, px, group, **style) -> None:
        self.queue.append((priority, len(self.queue), centers, text, px, style, group))

    def flush(self) -> None:
        """Place the queued labels by priority, one per group in each pass, so every line gets its share."""
        pending = sorted(self.queue, key=lambda item: (item[0], item[1]))
        while pending:
            placed, remaining = set(), []
            for item in pending:
                _, _, centers, text, px, style, group = item
                if group in placed:
                    remaining.append(item)
                elif self.put(centers, text, px, **style):
                    placed.add(group)
            pending = remaining
        self.queue = []


def _guide(canvas, labels, K, E, g_cm, axis, lo, hi, across, color, tint, px, priority=1.0) -> None:
    """A dashed line through ``g_cm`` along ``axis`` from ``lo`` to ``hi`` (cm), ticked every 5 cm and
    labelled every 10 cm except within 4 cm of the grasp point's own coordinate."""
    index = "xyz".index(axis)
    unit = np.eye(3)[index]

    def at(value):
        p = np.array(g_cm, dtype=float)
        p[index] = value
        return p / 100

    values = list(range(int(math.floor(lo)), int(math.floor(hi)) + 1))
    if hi > values[-1]:
        values.append(hi)
    points = {value: _project(K, E, at(value)) for value in values}
    canvas.line([points[value] for value in values], color, 1.3, halo=0.8, dash=(5, 4))
    for value in values:
        p = points[value]
        if round(value) % 5 or p is None:
            continue
        along, side = _image_direction(K, E, at(value), unit), _image_direction(K, E, at(value), across)
        if along is None or side is None:
            continue
        n = np.array([-along[1], along[0]])
        if n @ side < 0:
            n = -n
        ten = round(value) % 10 == 0
        length = 7.0 if ten else 4.0
        canvas.line([p, (p[0] + n[0] * length, p[1] + n[1] * length)], color, 1.5 if ten else 1.0, halo=0.8)
        if ten and abs(value - g_cm[index]) >= 4:
            if not (0 <= p[0] <= canvas.W and 0 <= p[1] <= canvas.H):
                continue
            centers = [(p[0] + n[0] * distance, p[1] + n[1] * distance)
                       for distance in (length + 9, -length - 9, length + 19, -length - 19)]
            labels.defer(priority, centers, f"{axis}={int(round(value))}", px, (color, axis), text_fill=tint,
                         anchor=p, leader_color=color)


def _legend(canvas, labels, K, E, centre, px) -> None:
    """Three short arrows in the least cluttered corner where they fit: the image directions of +x, +y, +z
    at ``centre``.  An axis seen end-on is drawn as a dot."""
    view = E[2, :3]
    directions = {}
    for name, axis in zip("xyz", np.eye(3)):
        a, b = _project(K, E, centre - axis * 0.025), _project(K, E, centre + axis * 0.025)
        if a is None or b is None:
            return
        d = np.subtract(b, a)
        end_on = abs(float(view @ axis)) > 0.985 or float(np.linalg.norm(d)) < 4
        directions[name] = (d / (np.linalg.norm(d) + 1e-9), end_on)
    inset = 60
    corners = [(inset, inset), (canvas.W - inset, inset), (inset, canvas.H - inset),
               (canvas.W - inset, canvas.H - inset)]
    ranked = sorted(enumerate(corners), key=lambda item: canvas.clutter(
        (item[1][0] - inset, item[1][1] - inset, item[1][0] + inset, item[1][1] + inset)) + item[0] * 1.5)
    for _, (ox, oy) in ranked:
        boxes, plan = [], []
        for name, (u, end_on) in directions.items():
            if end_on:
                others = sum(directions[other][0] for other in directions if other != name)
                away = np.asarray(-others / (np.linalg.norm(others) + 1e-9))
                plan.append((name, None, (ox + away[0] * 22, oy + away[1] * 22)))
            else:
                tip = (ox + u[0] * 26, oy + u[1] * 26)
                plan.append((name, tip, (tip[0] + u[0] * 14, tip[1] + u[1] * 14)))
            boxes.append(canvas.text_box(plan[-1][2], "+" + name, px))
        frame = (min(ox - 8, *(box[0] for box in boxes)), min(oy - 8, *(box[1] for box in boxes)),
                 max(ox + 8, *(box[2] for box in boxes)), max(oy + 8, *(box[3] for box in boxes)))
        if labels.fits(frame):
            break
    else:
        return
    origin = (ox, oy)
    for name, tip, center in plan:
        if tip is None:
            canvas.dot(origin, 7.0, AXIS[name], ring=(255, 255, 255), ring_width=1.5)
            canvas.dot(origin, 2.2, (255, 255, 255))
        else:
            canvas.arrow(origin, tip, AXIS[name])
        labels.put(center, "+" + name, px, fill=(*AXIS[name], 255))
    if all(tip is not None for _, tip, _ in plan):
        canvas.dot(origin, 3.5, (40, 40, 40), ring=(255, 255, 255), ring_width=1.2)
    labels.protected.append(frame)
