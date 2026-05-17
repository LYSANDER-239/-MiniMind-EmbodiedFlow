import json
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_jsonl(path: str | Path, limit: int | None = None) -> list[dict]:
    items = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if limit is not None and len(items) >= limit:
                break
    return items


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def save_jsonl(path: str | Path, items: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def resolve_image_path(labels_path: str | Path, image_value: str | Path) -> Path:
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return Path(labels_path).parent / image_path
