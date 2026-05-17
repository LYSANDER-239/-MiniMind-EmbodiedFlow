from pathlib import Path

from PIL import Image, ImageDraw

from core.io_utils import ensure_dir
from core.metrics import tensor_to_list


def _point_to_px(point, width: int, height: int) -> tuple[int, int]:
    return (
        int(round(float(point[0]) * (width - 1))),
        int(round(float(point[1]) * (height - 1))),
    )


def _box_to_px(box, width: int, height: int) -> list[int]:
    return [
        int(round(float(box[0]) * (width - 1))),
        int(round(float(box[1]) * (height - 1))),
        int(round(float(box[2]) * (width - 1))),
        int(round(float(box[3]) * (height - 1))),
    ]


def draw_overlay(
    image_path: str | Path,
    save_path: str | Path,
    detected: dict,
    grounding: dict,
    sample: dict,
    trajectory,
) -> str:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for obj in detected.get("objects", []):
        box = _box_to_px(obj["box"], width, height)
        draw.rectangle(box, outline=(220, 38, 38), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"{obj['color']}/{obj.get('shape', '?')}", fill=(220, 38, 38))

    for goal in detected.get("goals", []):
        box = _box_to_px(goal["box"], width, height)
        draw.rectangle(box, outline=(37, 99, 235), width=2)
        draw.text((box[0] + 2, max(0, box[1] - 12)), f"goal:{goal['color']}", fill=(37, 99, 235))

    for rect, valid in zip(grounding.get("obstacles", []), grounding.get("obstacle_mask", [])):
        if float(valid) > 0.5:
            draw.rectangle(_box_to_px(rect, width, height), outline=(194, 105, 28), width=3)

    points = [_point_to_px(point, width, height) for point in tensor_to_list(trajectory)]
    if len(points) >= 2:
        draw.line(points, fill=(37, 99, 235), width=4)

    if grounding.get("pred_start") is not None:
        x, y = _point_to_px(grounding["pred_start"], width, height)
        draw.rectangle([x - 5, y - 5, x + 5, y + 5], fill=(22, 163, 74))
    if grounding.get("pred_goal") is not None:
        x, y = _point_to_px(grounding["pred_goal"], width, height)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], outline=(147, 51, 234), width=3)

    # Ground-truth crosses are for visual QA only.
    for point, color in ((sample.get("start"), (220, 38, 38)), (sample.get("goal"), (37, 99, 235))):
        if point is None:
            continue
        x, y = _point_to_px(point, width, height)
        draw.line([x - 6, y - 6, x + 6, y + 6], fill=color, width=2)
        draw.line([x - 6, y + 6, x + 6, y - 6], fill=color, width=2)

    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    image.save(save_path)
    return str(save_path)


def make_summary_grid(image_paths: list[str | Path], save_path: str | Path, max_images: int = 16, thumb_size: int = 256) -> str | None:
    image_paths = [Path(path) for path in image_paths[:max_images] if Path(path).is_file()]
    if not image_paths:
        return None
    images = [Image.open(path).convert("RGB").resize((thumb_size, thumb_size)) for path in image_paths]
    cols = min(4, len(images))
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (255, 255, 255))
    for idx, image in enumerate(images):
        x = (idx % cols) * thumb_size
        y = (idx // cols) * thumb_size
        grid.paste(image, (x, y))
    save_path = Path(save_path)
    ensure_dir(save_path.parent)
    grid.save(save_path)
    return str(save_path)
