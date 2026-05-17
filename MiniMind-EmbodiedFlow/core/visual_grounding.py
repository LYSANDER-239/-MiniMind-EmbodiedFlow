from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BACKGROUND_RGB = np.array((245, 245, 245), dtype=np.float32)
OBJECT_RGB = {
    "red": (220, 50, 50),
    "yellow": (235, 200, 45),
    "green": (60, 170, 90),
    "purple": (145, 85, 210),
}
GOAL_RGB = {
    "blue": (55, 105, 235),
    "green": (80, 190, 105),
    "yellow": (245, 210, 65),
}
ORANGE_RGB = (240, 150, 60)
ORANGE_OUTLINE_RGB = (180, 95, 30)


def _blend_with_background(rgb, alpha: int = 80):
    color = np.array(rgb, dtype=np.float32)
    a = float(alpha) / 255.0
    return tuple(((1.0 - a) * BACKGROUND_RGB + a * color).round().astype(np.uint8).tolist())


GOAL_FILL_RGB = {color: _blend_with_background(rgb) for color, rgb in GOAL_RGB.items()}
GOAL_OUTLINE_RGB = {color: tuple(max(0, c - 85) for c in rgb) for color, rgb in GOAL_RGB.items()}


def color_distance_mask(image: np.ndarray, rgb, threshold: float) -> np.ndarray:
    target = np.array(rgb, dtype=np.float32)
    return np.linalg.norm(image.astype(np.float32) - target[None, None, :], axis=-1) < float(threshold)


def connected_components(mask: np.ndarray, min_area: int = 20) -> list[dict]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(x, y)])
            visited[y, x] = True
            xs = []
            ys = []
            while queue:
                cx, cy = queue.popleft()
                xs.append(cx)
                ys.append(cy)
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))

            area = len(xs)
            if area < min_area:
                continue
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            comp_mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=bool)
            comp_mask[np.array(ys) - y0, np.array(xs) - x0] = True
            components.append({
                "box_px": [x0, y0, x1, y1],
                "center_px": [float(np.mean(xs)), float(np.mean(ys))],
                "area": int(area),
                "mask": comp_mask,
            })
    return components


def _norm_point(point_px, width: int, height: int) -> list[float]:
    return [float(point_px[0]) / max(width - 1, 1), float(point_px[1]) / max(height - 1, 1)]


def _norm_box(box_px, width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box_px
    return [
        float(x0) / max(width - 1, 1),
        float(y0) / max(height - 1, 1),
        float(x1) / max(width - 1, 1),
        float(y1) / max(height - 1, 1),
    ]


def classify_shape(component: dict) -> str:
    x0, y0, x1, y1 = component["box_px"]
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    fill_ratio = float(component["area"]) / float(w * h)
    mask = component["mask"]
    ch = max(1, int(round(h * 0.25)))
    cw = max(1, int(round(w * 0.25)))
    corner_area = 4 * ch * cw
    corner_count = (
        int(mask[:ch, :cw].sum())
        + int(mask[:ch, -cw:].sum())
        + int(mask[-ch:, :cw].sum())
        + int(mask[-ch:, -cw:].sum())
    )
    corner_fill_ratio = corner_count / max(corner_area, 1)

    if fill_ratio < 0.62:
        return "triangle"
    if corner_fill_ratio > 0.55 or fill_ratio > 0.86:
        return "square"
    return "circle"


def _component_record(component: dict, color: str, width: int, height: int, include_shape: bool = False) -> dict:
    record = {
        "color": color,
        "center": _norm_point(component["center_px"], width, height),
        "box": _norm_box(component["box_px"], width, height),
        "area": int(component["area"]),
        "confidence": 1.0,
    }
    if include_shape:
        record["shape"] = classify_shape(component)
    return record


def _merge_goal_components(components: list[dict]) -> list[dict]:
    if not components:
        return []
    components = sorted(components, key=lambda c: c["area"], reverse=True)
    # Goal fill and outline sometimes split under strict thresholds. The largest
    # component is the region fill and gives the most stable center.
    return [components[0]]


def _pad_obstacles(obstacles: list[list[float]], max_obstacles: int) -> tuple[list[list[float]], list[float]]:
    obstacles = obstacles[:max_obstacles]
    mask = [1.0] * len(obstacles)
    while len(obstacles) < max_obstacles:
        obstacles.append([0.0, 0.0, 0.0, 0.0])
        mask.append(0.0)
    return obstacles, mask


def detect_scene_elements_from_image(
    image_path: str,
    image_size: int = 256,
    color_threshold: float = 45.0,
    goal_color_threshold: float = 40.0,
    obstacle_color_threshold: float = 70.0,
    min_component_area: int = 20,
    min_goal_area: int = 120,
    max_obstacles: int = 2,
) -> dict:
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    height, width = arr.shape[:2]

    objects = []
    for color, rgb in OBJECT_RGB.items():
        mask = color_distance_mask(arr, rgb, color_threshold)
        for component in connected_components(mask, min_area=min_component_area):
            objects.append(_component_record(component, color, width, height, include_shape=True))

    goals = []
    for color in GOAL_RGB:
        fill_mask = color_distance_mask(arr, GOAL_FILL_RGB[color], goal_color_threshold)
        outline_mask = color_distance_mask(arr, GOAL_OUTLINE_RGB[color], goal_color_threshold * 0.9)
        mask = fill_mask | outline_mask
        # Remove solid object pixels of the same color family; goals are larger,
        # pale translucent regions in these synthetic scenes.
        if color in OBJECT_RGB:
            mask &= ~color_distance_mask(arr, OBJECT_RGB[color], color_threshold)
        components = [c for c in connected_components(mask, min_area=min_component_area) if c["area"] >= min_goal_area]
        for component in _merge_goal_components(components):
            goals.append(_component_record(component, color, width, height, include_shape=False))

    obstacle_mask = (
        color_distance_mask(arr, ORANGE_RGB, obstacle_color_threshold)
        | color_distance_mask(arr, ORANGE_OUTLINE_RGB, obstacle_color_threshold)
    )
    obstacle_components = connected_components(obstacle_mask, min_area=min_component_area)
    obstacle_components = sorted(obstacle_components, key=lambda c: c["area"], reverse=True)
    obstacles = [_norm_box(component["box_px"], width, height) for component in obstacle_components[:max_obstacles]]
    obstacles, obstacle_valid = _pad_obstacles(obstacles, max_obstacles)

    return {
        "objects": objects,
        "goals": goals,
        "obstacles": obstacles,
        "obstacle_mask": obstacle_valid,
    }


def ground_from_visual_detection(parsed: dict, detected: dict, max_obstacles: int = 2) -> dict:
    target_matches = [
        obj for obj in detected.get("objects", [])
        if obj.get("color") == parsed.get("target_color") and obj.get("shape") == parsed.get("target_shape")
    ]
    if not target_matches:
        return {
            "success": False,
            "target_object": None,
            "goal": None,
            "pred_start": None,
            "pred_goal": None,
            "obstacles": detected.get("obstacles", [])[:max_obstacles],
            "obstacle_mask": detected.get("obstacle_mask", [])[:max_obstacles],
            "error": "target_object_not_found",
        }
    if len(target_matches) > 1:
        return {
            "success": False,
            "target_object": None,
            "goal": None,
            "pred_start": None,
            "pred_goal": None,
            "obstacles": detected.get("obstacles", [])[:max_obstacles],
            "obstacle_mask": detected.get("obstacle_mask", [])[:max_obstacles],
            "error": "target_object_ambiguous",
        }

    goal_matches = [goal for goal in detected.get("goals", []) if goal.get("color") == parsed.get("goal_color")]
    if not goal_matches:
        return {
            "success": False,
            "target_object": target_matches[0],
            "goal": None,
            "pred_start": target_matches[0]["center"],
            "pred_goal": None,
            "obstacles": detected.get("obstacles", [])[:max_obstacles],
            "obstacle_mask": detected.get("obstacle_mask", [])[:max_obstacles],
            "error": "goal_not_found",
        }
    if len(goal_matches) > 1:
        return {
            "success": False,
            "target_object": target_matches[0],
            "goal": None,
            "pred_start": target_matches[0]["center"],
            "pred_goal": None,
            "obstacles": detected.get("obstacles", [])[:max_obstacles],
            "obstacle_mask": detected.get("obstacle_mask", [])[:max_obstacles],
            "error": "goal_ambiguous",
        }

    obstacles, obstacle_mask = _pad_obstacles([list(rect) for rect in detected.get("obstacles", [])], max_obstacles)
    if not obstacles or len(obstacle_mask) != max_obstacles:
        error = "obstacle_detection_failed"
    else:
        error = None
    return {
        "success": error is None,
        "target_object": target_matches[0],
        "goal": goal_matches[0],
        "pred_start": target_matches[0]["center"],
        "pred_goal": goal_matches[0]["center"],
        "obstacles": obstacles,
        "obstacle_mask": obstacle_mask,
        "error": error,
    }


def rect_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = map(float, a)
    bx0, by0, bx1, by1 = map(float, b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def draw_detection_overlay(
    image_path: str | Path,
    save_path: str | Path,
    detected: dict,
    grounding: dict | None = None,
    sample: dict | None = None,
    show_gt: bool = True,
):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    def to_px_box(box):
        return [
            int(round(float(box[0]) * (width - 1))),
            int(round(float(box[1]) * (height - 1))),
            int(round(float(box[2]) * (width - 1))),
            int(round(float(box[3]) * (height - 1))),
        ]

    def to_px_point(point):
        return (
            int(round(float(point[0]) * (width - 1))),
            int(round(float(point[1]) * (height - 1))),
        )

    for obj in detected.get("objects", []):
        box = to_px_box(obj["box"])
        draw.rectangle(box, outline=(220, 38, 38), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"{obj['color']}/{obj.get('shape', '?')}", fill=(220, 38, 38))
    for goal in detected.get("goals", []):
        box = to_px_box(goal["box"])
        draw.rectangle(box, outline=(37, 99, 235), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"goal:{goal['color']}", fill=(37, 99, 235))
    for rect, valid in zip(detected.get("obstacles", []), detected.get("obstacle_mask", [])):
        if float(valid) > 0.5:
            draw.rectangle(to_px_box(rect), outline=(194, 105, 28), width=3)

    if grounding and grounding.get("pred_start") is not None:
        x, y = to_px_point(grounding["pred_start"])
        draw.rectangle([x - 5, y - 5, x + 5, y + 5], outline=(22, 163, 74), width=3)
    if grounding and grounding.get("pred_goal") is not None:
        x, y = to_px_point(grounding["pred_goal"])
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(147, 51, 234), width=3)

    if show_gt and sample is not None:
        for point, color in ((sample.get("start"), (0, 120, 0)), (sample.get("goal"), (80, 0, 160))):
            if point is None:
                continue
            x, y = to_px_point(point)
            draw.line([x - 6, y - 6, x + 6, y + 6], fill=color, width=2)
            draw.line([x - 6, y + 6, x + 6, y - 6], fill=color, width=2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(save_path)
