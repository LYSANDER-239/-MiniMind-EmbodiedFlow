import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _default_conda_python() -> Path:
    return Path.home() / ".conda" / "envs" / "minimind-v" / ("python.exe" if os.name == "nt" else "bin/python")


def resolve_python_executable(value: str) -> str:
    if value != "auto":
        return str(Path(value))
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = Path(conda_prefix) / ("python.exe" if os.name == "nt" else "bin/python")
        if candidate.is_file():
            return str(candidate)
    candidate = _default_conda_python()
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def configure_runtime_env(env: dict) -> dict:
    env = env.copy()
    tmp_dir = PROJECT_ROOT / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env["TMP"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMPDIR"] = str(tmp_dir)
    site_dir = PROJECT_ROOT / "tmp_python_site"
    python_path = [str(PROJECT_ROOT)]
    if site_dir.is_dir():
        python_path.append(str(site_dir))
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Run the minimal MiniMind-EmbodiedFlow demo.")
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--minimind_v_root", type=str, required=True)
    parser.add_argument("--vlm_checkpoint", type=str, required=True)
    parser.add_argument("--flow_checkpoint", type=str, default="outputs/two_obstacle_flow_mlp.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/final_demo")
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--success_end_threshold", type=float, default=0.08)
    parser.add_argument("--center_threshold", type=float, default=0.05)
    parser.add_argument("--obstacle_iou_threshold", type=float, default=0.5)
    parser.add_argument("--python_executable", type=str, default="auto")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def maybe_relaunch(args) -> None:
    if args._child:
        return
    python_executable = Path(resolve_python_executable(args.python_executable))
    if python_executable.resolve() == Path(sys.executable).resolve():
        return
    command = [str(python_executable), str(Path(__file__).resolve()), *sys.argv[1:], "--_child"]
    print("Relaunching final demo with:", python_executable)
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT), env=configure_runtime_env(os.environ))
    raise SystemExit(0)


def _device_from_arg(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _first_error(parser_result, parse_compare, grounding, target_ok, goal_ok, obstacle_iou, traj_stats, obstacle_iou_threshold):
    if not parser_result.get("json_valid", False):
        return "json_invalid"
    if not parser_result.get("parse_valid", False):
        return parser_result.get("error") or "parse_invalid"
    if not parse_compare.get("exact_match", False):
        return "parse_exact_match_failed"
    if not grounding.get("success", False):
        return grounding.get("error") or "visual_grounding_failed"
    if not target_ok:
        return "target_object_wrong"
    if not goal_ok:
        return "goal_wrong"
    if obstacle_iou < obstacle_iou_threshold:
        return "obstacle_iou_low"
    if traj_stats.get("collision"):
        return "trajectory_collision"
    if not traj_stats.get("success", False):
        return "trajectory_end_error_too_high"
    return None


def run_demo(args) -> None:
    import torch

    from core.flow_sampler import (
        build_condition,
        build_flow_batch_from_groundings,
        load_flow_model,
        repair_colliding_trajectories,
        sample_trajectories_with_candidates,
    )
    from core.io_utils import ensure_dir, load_jsonl, resolve_image_path, save_json, save_jsonl
    from core.metrics import (
        compare_parse,
        compute_trajectory_metrics,
        is_parse_valid,
        l2,
        mean,
        obstacle_iou_score,
        tensor_to_list,
        valid_rects,
    )
    from core.minimind_parser import MiniMindVParser
    from core.visual_grounding import detect_scene_elements_from_image, ground_from_visual_detection
    from core.visualization import draw_overlay, make_summary_grid

    torch.manual_seed(args.seed)
    device = _device_from_arg(args.device)
    labels_path = Path(args.labels)
    output_dir = ensure_dir(args.output_dir)
    overlays_dir = ensure_dir(output_dir / "overlays")

    samples = load_jsonl(labels_path, args.num_samples)
    parser = MiniMindVParser(
        minimind_v_root=args.minimind_v_root,
        model_path=args.vlm_checkpoint,
        device=str(device),
        prompt_mode="zero_shot",
    )

    parser_results = []
    parse_compares = []
    detections = []
    groundings = []
    raw_outputs = []
    for sample in samples:
        image_path = resolve_image_path(labels_path, sample["image"])
        instruction = sample["task"]["instruction"]
        result = parser.parse(str(image_path), instruction)
        pred_parse = result["pred_parse"]
        detected = detect_scene_elements_from_image(str(image_path))
        grounding = ground_from_visual_detection(pred_parse, detected)
        parser_results.append(result)
        parse_compares.append(compare_parse(pred_parse, sample["task"]["parse_answer"]))
        detections.append(detected)
        groundings.append(grounding)
        raw_outputs.append({
            "scene_id": sample["scene_id"],
            "instruction": instruction,
            "prompt": result.get("prompt"),
            "raw_output": result.get("raw_output"),
        })

    flow_model, _flow_config, _mode, traj_len = load_flow_model(args.flow_checkpoint, device)
    flow_batch = build_flow_batch_from_groundings(samples, groundings, int(flow_model.condition_dim), device)
    condition = build_condition(flow_batch)
    generated, _raw_generated, selected_candidate = sample_trajectories_with_candidates(
        model=flow_model,
        condition=condition,
        batch=flow_batch,
        traj_len=traj_len,
        num_steps=args.num_steps,
        num_candidates=args.num_candidates,
        device=device,
        rerank_clearance=0.06,
    )
    generated = repair_colliding_trajectories(generated, flow_batch, clearance=0.04)

    results = []
    errors = []
    overlay_paths = []
    for idx, sample in enumerate(samples):
        gt_parse = sample["task"]["parse_answer"]
        pred_parse = parser_results[idx]["pred_parse"]
        grounding = groundings[idx]
        gt_obstacles = valid_rects(sample.get("obstacles", []), sample.get("obstacle_mask", []))
        target_ok = l2(grounding.get("pred_start"), sample.get("start")) <= args.center_threshold
        goal_ok = l2(grounding.get("pred_goal"), sample.get("goal")) <= args.center_threshold
        obstacle_iou = obstacle_iou_score(
            grounding.get("obstacles", []),
            grounding.get("obstacle_mask", []),
            gt_obstacles,
        )
        traj_stats = compute_trajectory_metrics(
            generated[idx],
            grounding.get("pred_start"),
            grounding.get("pred_goal"),
            gt_obstacles,
            success_end_threshold=args.success_end_threshold,
        )
        traj_stats["success"] = bool(
            parser_results[idx].get("parse_valid", False)
            and grounding.get("success", False)
            and traj_stats["success"]
        )

        image_path = resolve_image_path(labels_path, sample["image"])
        overlay_rel = Path("overlays") / f"scene_{int(sample['scene_id']):06d}_overlay.png"
        overlay_path = output_dir / overlay_rel
        draw_overlay(image_path, overlay_path, detections[idx], grounding, sample, generated[idx])
        overlay_paths.append(str(overlay_path))

        error = _first_error(
            parser_results[idx],
            parse_compares[idx],
            grounding,
            target_ok,
            goal_ok,
            obstacle_iou,
            traj_stats,
            args.obstacle_iou_threshold,
        )
        record = {
            "scene_id": sample["scene_id"],
            "image": sample["image"],
            "instruction": sample["task"]["instruction"],
            "gt_parse": gt_parse,
            "raw_output": parser_results[idx].get("raw_output"),
            "pred_parse": pred_parse,
            "json_valid": bool(parser_results[idx].get("json_valid", False)),
            "parse_valid": bool(is_parse_valid(pred_parse)),
            "parse_exact_match": bool(parse_compares[idx]["exact_match"]),
            "gt_target_object_id": sample["task"]["target_object_id"],
            "gt_goal_id": sample["task"]["goal_id"],
            "gt_start": sample["start"],
            "pred_start": grounding.get("pred_start"),
            "gt_goal": sample["goal"],
            "pred_goal": grounding.get("pred_goal"),
            "obstacles": grounding.get("obstacles", []),
            "obstacle_mask": grounding.get("obstacle_mask", []),
            "obstacle_iou": obstacle_iou,
            "selected_candidate": int(selected_candidate[idx].detach().cpu()),
            "trajectory": tensor_to_list(generated[idx]),
            "trajectory_metrics": traj_stats,
            "overlay_path": str(overlay_rel).replace("\\", "/"),
            "error": error,
        }
        results.append(record)
        if error:
            errors.append({
                "scene_id": sample["scene_id"],
                "instruction": sample["task"]["instruction"],
                "gt_parse": gt_parse,
                "raw_output": parser_results[idx].get("raw_output"),
                "pred_parse": pred_parse,
                "grounding": grounding,
                "trajectory_metrics": traj_stats,
                "error": error,
            })

    metrics = {
        "num_samples": len(samples),
        "parser": {
            "json_valid_rate": mean([float(r.get("json_valid", False)) for r in parser_results]),
            "parse_valid_rate": mean([float(r.get("parse_valid", False)) for r in parser_results]),
            "exact_match": mean([float(c["exact_match"]) for c in parse_compares]),
            "target_color_acc": mean([float(c["target_color"]) for c in parse_compares]),
            "target_shape_acc": mean([float(c["target_shape"]) for c in parse_compares]),
            "goal_color_acc": mean([float(c["goal_color"]) for c in parse_compares]),
            "action_type_acc": mean([float(c["action_type"]) for c in parse_compares]),
            "avoid_color_acc": mean([float(c["avoid_color"]) for c in parse_compares]),
        },
        "visual_grounding": {
            "grounding_success_rate": mean([float(g.get("success", False)) for g in groundings]),
            "target_object_acc": mean([float(l2(g.get("pred_start"), s.get("start")) <= args.center_threshold) for g, s in zip(groundings, samples)]),
            "goal_acc": mean([float(l2(g.get("pred_goal"), s.get("goal")) <= args.center_threshold) for g, s in zip(groundings, samples)]),
            "start_l2_error": mean([l2(g.get("pred_start"), s.get("start")) for g, s in zip(groundings, samples)]),
            "goal_l2_error": mean([l2(g.get("pred_goal"), s.get("goal")) for g, s in zip(groundings, samples)]),
            "obstacle_iou_mean": mean([float(r["obstacle_iou"]) for r in results]),
        },
        "trajectory": {
            "start_error": mean([r["trajectory_metrics"]["start_error"] for r in results]),
            "end_error": mean([r["trajectory_metrics"]["end_error"] for r in results]),
            "collision_rate": mean([float(r["trajectory_metrics"]["collision"]) for r in results]),
            "success_rate": mean([float(r["trajectory_metrics"]["success"]) for r in results]),
            "path_length": mean([r["trajectory_metrics"]["path_length"] for r in results]),
            "smoothness": mean([r["trajectory_metrics"]["smoothness"] for r in results]),
        },
        "num_error_cases": len(errors),
    }

    save_json(output_dir / "metrics.json", metrics)
    save_jsonl(output_dir / "results.jsonl", results)
    save_jsonl(output_dir / "raw_outputs.jsonl", raw_outputs)
    save_jsonl(output_dir / "error_cases.jsonl", errors)
    make_summary_grid(overlay_paths, output_dir / "summary_grid.png")
    print(f"Final demo written to {output_dir}")
    print(f"success_rate={metrics['trajectory']['success_rate']:.3f}, collision_rate={metrics['trajectory']['collision_rate']:.3f}")


def main():
    args = parse_args()
    maybe_relaunch(args)
    run_demo(args)


if __name__ == "__main__":
    main()
