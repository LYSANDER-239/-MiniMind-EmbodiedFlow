import math

import torch

from core.flow_sampler import trajectory_collides_with_rect, trajectory_path_length, trajectory_smoothness


PARSE_FIELDS = ["target_color", "target_shape", "goal_color", "action_type", "avoid_color"]


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def l2(a, b) -> float:
    if a is None or b is None:
        return math.inf
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def tensor_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def is_parse_valid(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    if parsed.get("error"):
        return False
    return all(parsed.get(field) is not None for field in PARSE_FIELDS)


def compare_parse(pred: dict, gt: dict) -> dict:
    result = {}
    for field in PARSE_FIELDS:
        result[field] = pred.get(field) == gt.get(field)
    result["exact_match"] = all(result[field] for field in PARSE_FIELDS)
    return result


def valid_rects(obstacles: list, obstacle_mask: list) -> list[list[float]]:
    return [rect for rect, valid in zip(obstacles, obstacle_mask) if float(valid) > 0.5]


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


def obstacle_iou_score(pred_obstacles, pred_mask, gt_obstacles) -> float:
    pred_valid = valid_rects(pred_obstacles, pred_mask)
    if not gt_obstacles and not pred_valid:
        return 1.0
    if not gt_obstacles or not pred_valid:
        return 0.0
    used = set()
    scores = []
    for gt in gt_obstacles:
        best_iou = 0.0
        best_idx = None
        for idx, pred in enumerate(pred_valid):
            if idx in used:
                continue
            iou = rect_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
        scores.append(best_iou)
    return mean(scores)


def compute_trajectory_metrics(
    trajectory,
    pred_start,
    pred_goal,
    gt_obstacles,
    success_end_threshold: float = 0.08,
) -> dict:
    traj = trajectory.detach().cpu() if isinstance(trajectory, torch.Tensor) else torch.tensor(trajectory, dtype=torch.float32)
    start_error = l2(traj[0].tolist(), pred_start)
    end_error = l2(traj[-1].tolist(), pred_goal)
    collision = any(trajectory_collides_with_rect(traj, rect, clearance=0.0) for rect in gt_obstacles)
    return {
        "start_error": start_error,
        "end_error": end_error,
        "collision": bool(collision),
        "path_length": float(trajectory_path_length(traj)),
        "smoothness": float(trajectory_smoothness(traj)),
        "success": bool((not collision) and end_error < success_end_threshold),
    }
