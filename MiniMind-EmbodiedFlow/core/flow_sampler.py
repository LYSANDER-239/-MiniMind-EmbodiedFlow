import heapq
import math

import torch

from core.flow_model import ConditionalFlowMLP


MODE = "multi_obstacle"
EPS = 1e-8


def normalize_mode(mode: str) -> str:
    return "fixed_obstacle" if mode == "obstacle" else mode


def condition_dim_for_mode(mode: str, max_obstacles: int = 2) -> int:
    if normalize_mode(mode) == "multi_obstacle":
        return 4 + max_obstacles * 5
    return 4


def infer_max_obstacles(condition_dim: int) -> int:
    if (condition_dim - 4) % 5 != 0:
        raise ValueError(f"Cannot infer max_obstacles from condition_dim={condition_dim}")
    return (condition_dim - 4) // 5


def pad_obstacles(obstacles: list, obstacle_mask: list, target_count: int):
    padded_obstacles = [list(map(float, rect)) for rect in obstacles]
    padded_mask = [float(v) for v in obstacle_mask]
    if len(padded_obstacles) > target_count:
        padded_obstacles = padded_obstacles[:target_count]
        padded_mask = padded_mask[:target_count]
    while len(padded_obstacles) < target_count:
        padded_obstacles.append([0.0, 0.0, 0.0, 0.0])
        padded_mask.append(0.0)
    return padded_obstacles, padded_mask


def build_condition(batch: dict, mode: str = MODE) -> torch.Tensor:
    if normalize_mode(mode) != "multi_obstacle":
        return torch.cat([batch["start"], batch["goal"]], dim=-1)
    obstacles_flat = batch["obstacles"].reshape(batch["obstacles"].shape[0], -1)
    return torch.cat([batch["start"], batch["goal"], obstacles_flat, batch["obstacle_mask"]], dim=-1)


def build_flow_batch_from_groundings(samples: list[dict], groundings: list[dict], condition_dim: int, device):
    max_obstacles = infer_max_obstacles(condition_dim)
    starts, goals, obstacles_all, masks_all = [], [], [], []
    for sample, grounding in zip(samples, groundings):
        if grounding["success"]:
            starts.append(grounding["pred_start"])
            goals.append(grounding["pred_goal"])
            obstacles, mask = pad_obstacles(
                grounding.get("obstacles", []),
                grounding.get("obstacle_mask", []),
                max_obstacles,
            )
        else:
            starts.append(sample["start"])
            goals.append(sample["goal"])
            obstacles, mask = pad_obstacles([], [], max_obstacles)
        obstacles_all.append(obstacles)
        masks_all.append(mask)
    return {
        "start": torch.tensor(starts, dtype=torch.float32, device=device),
        "goal": torch.tensor(goals, dtype=torch.float32, device=device),
        "obstacles": torch.tensor(obstacles_all, dtype=torch.float32, device=device),
        "obstacle_mask": torch.tensor(masks_all, dtype=torch.float32, device=device),
    }


def load_flow_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    mode = normalize_mode(config.get("mode", "multi_obstacle"))
    traj_len = ckpt.get("traj_len", config.get("traj_len", 32))
    condition_dim = ckpt.get("condition_dim", condition_dim_for_mode(mode, config.get("max_obstacles", 2)))
    model = ConditionalFlowMLP(
        traj_len=traj_len,
        condition_dim=condition_dim,
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, mode, traj_len


@torch.no_grad()
def sample_trajectories(model, condition, start, goal, traj_len: int, num_steps: int, device) -> torch.Tensor:
    batch_size = condition.shape[0]
    z = torch.randn(batch_size, traj_len, 2, device=device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size, 1), step / num_steps, device=device)
        z = z + dt * model(z_t=z, t=t, condition=condition)
        z[:, 0, :] = start
        z[:, -1, :] = goal
    return z


def expand_rect(rect, margin: float):
    xmin, ymin, xmax, ymax = [float(v) for v in rect]
    return max(0.0, xmin - margin), max(0.0, ymin - margin), min(1.0, xmax + margin), min(1.0, ymax + margin)


def point_in_rect(point, rect, clearance: float = 0.0) -> bool:
    x, y = float(point[0]), float(point[1])
    xmin, ymin, xmax, ymax = expand_rect(rect, clearance)
    return xmin <= x <= xmax and ymin <= y <= ymax


def _orientation(a, b, c) -> int:
    val = (float(b[1]) - float(a[1])) * (float(c[0]) - float(b[0])) - (
        float(b[0]) - float(a[0])
    ) * (float(c[1]) - float(b[1]))
    if abs(val) < 1e-10:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a, b, c) -> bool:
    return (
        min(float(a[0]), float(c[0])) - 1e-10 <= float(b[0]) <= max(float(a[0]), float(c[0])) + 1e-10
        and min(float(a[1]), float(c[1])) - 1e-10 <= float(b[1]) <= max(float(a[1]), float(c[1])) + 1e-10
    )


def segments_intersect(p1, p2, p3, p4) -> bool:
    o1 = _orientation(p1, p2, p3)
    o2 = _orientation(p1, p2, p4)
    o3 = _orientation(p3, p4, p1)
    o4 = _orientation(p3, p4, p2)
    return (
        (o1 != o2 and o3 != o4)
        or (o1 == 0 and _on_segment(p1, p3, p2))
        or (o2 == 0 and _on_segment(p1, p4, p2))
        or (o3 == 0 and _on_segment(p3, p1, p4))
        or (o4 == 0 and _on_segment(p3, p2, p4))
    )


def line_segment_intersects_rect(p1, p2, rect, clearance: float = 0.0) -> bool:
    xmin, ymin, xmax, ymax = expand_rect(rect, clearance)
    if point_in_rect(p1, (xmin, ymin, xmax, ymax)) or point_in_rect(p2, (xmin, ymin, xmax, ymax)):
        return True
    if (float(p1[0]) < xmin and float(p2[0]) < xmin) or (float(p1[0]) > xmax and float(p2[0]) > xmax):
        return False
    if (float(p1[1]) < ymin and float(p2[1]) < ymin) or (float(p1[1]) > ymax and float(p2[1]) > ymax):
        return False
    edges = [((xmin, ymin), (xmax, ymin)), ((xmax, ymin), (xmax, ymax)), ((xmax, ymax), (xmin, ymax)), ((xmin, ymax), (xmin, ymin))]
    return any(segments_intersect(p1, p2, a, b) for a, b in edges)


def trajectory_collides_with_rect(traj: torch.Tensor, rect, clearance: float = 0.0) -> bool:
    for point in traj:
        if point_in_rect(point, rect, clearance):
            return True
    for i in range(traj.shape[0] - 1):
        if line_segment_intersects_rect(traj[i], traj[i + 1], rect, clearance):
            return True
    return False


def rects_for_sample(batch: dict, sample_idx: int):
    return [
        rect.detach().cpu()
        for rect, valid in zip(batch["obstacles"][sample_idx], batch["obstacle_mask"][sample_idx])
        if float(valid) > 0.5
    ]


def trajectory_collision_score(traj: torch.Tensor, rects, clearance: float = 0.0) -> tuple[int, int]:
    traj_cpu = traj.detach().cpu()
    point_hits = 0
    segment_hits = 0
    for rect in rects:
        point_hits += sum(point_in_rect(p, rect, clearance=clearance) for p in traj_cpu)
        segment_hits += sum(
            line_segment_intersects_rect(traj_cpu[i], traj_cpu[i + 1], rect, clearance=clearance)
            for i in range(traj_cpu.shape[0] - 1)
        )
    return int(point_hits), int(segment_hits)


def trajectory_path_length(traj: torch.Tensor) -> torch.Tensor:
    return torch.norm(traj[1:] - traj[:-1], dim=-1).sum()


def trajectory_smoothness(traj: torch.Tensor) -> torch.Tensor:
    if traj.shape[0] < 3:
        return traj.new_tensor(0.0)
    second = traj[2:] - 2.0 * traj[1:-1] + traj[:-2]
    return torch.norm(second, dim=-1).mean()


@torch.no_grad()
def sample_trajectories_with_candidates(
    model,
    condition,
    batch,
    traj_len: int,
    num_steps: int,
    num_candidates: int,
    device,
    rerank_clearance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_candidates <= 1:
        generated = sample_trajectories(model, condition, batch["start"], batch["goal"], traj_len, num_steps, device)
        return generated, generated, torch.zeros(generated.shape[0], dtype=torch.long, device=device)

    batch_size = condition.shape[0]
    condition_rep = condition.repeat_interleave(num_candidates, dim=0)
    start_rep = batch["start"].repeat_interleave(num_candidates, dim=0)
    goal_rep = batch["goal"].repeat_interleave(num_candidates, dim=0)
    all_generated = sample_trajectories(
        model, condition_rep, start_rep, goal_rep, traj_len, num_steps, device
    ).view(batch_size, num_candidates, traj_len, 2)

    selected = []
    selected_indices = []
    for i in range(batch_size):
        rects = rects_for_sample(batch, i)
        scored = []
        for c in range(num_candidates):
            traj = all_generated[i, c]
            point_hits, segment_hits = trajectory_collision_score(traj, rects, clearance=rerank_clearance)
            length = float(trajectory_path_length(traj.detach().cpu()))
            smooth = float(trajectory_smoothness(traj.detach().cpu()))
            score = (10000 * int(segment_hits > 0 or point_hits > 0), 100 * segment_hits + point_hits, length + 0.15 * smooth)
            scored.append((score, c))
        scored.sort(key=lambda item: item[0])
        best_idx = scored[0][1]
        selected.append(all_generated[i, best_idx])
        selected_indices.append(best_idx)
    return torch.stack(selected, dim=0), all_generated[:, 0], torch.tensor(selected_indices, dtype=torch.long, device=device)


def _resample_polyline(polyline: torch.Tensor, num_points: int) -> torch.Tensor:
    diffs = polyline[1:] - polyline[:-1]
    seg_len = torch.norm(diffs, dim=-1)
    cumlen = torch.cat([torch.zeros(1, device=polyline.device), torch.cumsum(seg_len, dim=0)])
    total = cumlen[-1]
    if total < EPS:
        return polyline[:1].expand(num_points, -1).clone()
    target = torch.linspace(0.0, float(total), num_points, device=polyline.device)
    idx = torch.searchsorted(cumlen, target, right=True) - 1
    idx = torch.clamp(idx, 0, polyline.shape[0] - 2)
    frac = ((target - cumlen[idx]) / (cumlen[idx + 1] - cumlen[idx]).clamp(min=EPS)).unsqueeze(-1)
    return polyline[idx] + frac * (polyline[idx + 1] - polyline[idx])


def _polyline_clear(polyline: list[torch.Tensor], obstacles, clearance: float) -> bool:
    for i in range(len(polyline) - 1):
        for rect in obstacles:
            if line_segment_intersects_rect(polyline[i], polyline[i + 1], rect, clearance):
                return False
    return True


def generate_visibility_graph_trajectory(start: torch.Tensor, goal: torch.Tensor, obstacles, traj_len: int, clearance: float = 0.04) -> torch.Tensor:
    if all(not line_segment_intersects_rect(start, goal, rect, clearance) for rect in obstacles):
        return _resample_polyline(torch.stack([start, goal]), traj_len)
    nodes = [start, goal]
    for rect in obstacles:
        xmin, ymin, xmax, ymax = expand_rect(rect, clearance)
        eps = 1e-3
        corners = [
            torch.tensor([max(0.0, xmin - eps), max(0.0, ymin - eps)], dtype=start.dtype),
            torch.tensor([max(0.0, xmin - eps), min(1.0, ymax + eps)], dtype=start.dtype),
            torch.tensor([min(1.0, xmax + eps), max(0.0, ymin - eps)], dtype=start.dtype),
            torch.tensor([min(1.0, xmax + eps), min(1.0, ymax + eps)], dtype=start.dtype),
        ]
        nodes.extend(c.clamp(0.0, 1.0) for c in corners)
    graph = [[] for _ in nodes]
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _polyline_clear([nodes[i], nodes[j]], obstacles, clearance):
                dist = float(torch.norm(nodes[i] - nodes[j]))
                graph[i].append((j, dist))
                graph[j].append((i, dist))
    heap = [(0.0, 0, [])]
    best = {0: 0.0}
    while heap:
        cost, idx, path = heapq.heappop(heap)
        if idx == 1:
            poly = [nodes[k] for k in path + [idx]]
            return _resample_polyline(torch.stack(poly), traj_len)
        for nxt, edge_cost in graph[idx]:
            new_cost = cost + edge_cost
            if new_cost < best.get(nxt, math.inf):
                best[nxt] = new_cost
                heapq.heappush(heap, (new_cost, nxt, path + [idx]))
    return _resample_polyline(torch.stack([start, goal]), traj_len)


def repair_colliding_trajectories(generated: torch.Tensor, batch: dict, clearance: float = 0.04) -> torch.Tensor:
    repaired = generated.detach().clone().cpu()
    starts = batch["start"].cpu()
    goals = batch["goal"].cpu()
    for i in range(repaired.shape[0]):
        rects = rects_for_sample({k: v.cpu() for k, v in batch.items()}, i)
        if rects and any(trajectory_collides_with_rect(repaired[i], rect, clearance=0.0) for rect in rects):
            repaired[i] = generate_visibility_graph_trajectory(starts[i], goals[i], rects, repaired.shape[1], clearance=clearance)
    return repaired.to(generated.device)
