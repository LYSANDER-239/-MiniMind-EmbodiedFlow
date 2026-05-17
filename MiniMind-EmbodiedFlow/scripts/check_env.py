import argparse
import json
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
    parts = [str(PROJECT_ROOT)]
    if site_dir.is_dir():
        parts.append(str(site_dir))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def parse_args():
    parser = argparse.ArgumentParser(description="Check the minimal MiniMind-EmbodiedFlow runtime environment.")
    parser.add_argument("--minimind_v_root", type=str, required=True)
    parser.add_argument("--vlm_checkpoint", type=str, required=True)
    parser.add_argument("--flow_checkpoint", type=str, required=True)
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--python_executable", type=str, default="auto")
    parser.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def maybe_relaunch(args):
    if args._child:
        return
    python_executable = Path(resolve_python_executable(args.python_executable))
    if python_executable.resolve() == Path(sys.executable).resolve():
        return
    command = [str(python_executable), str(Path(__file__).resolve()), *sys.argv[1:], "--_child"]
    print("Relaunching env check with:", python_executable)
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT), env=configure_runtime_env(os.environ))
    raise SystemExit(0)


def _try_import(name: str) -> dict:
    try:
        module = __import__(name)
        return {"ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def main():
    args = parse_args()
    maybe_relaunch(args)

    report = {
        "python": sys.executable,
        "paths": {
            "minimind_v_root": str(Path(args.minimind_v_root)),
            "vlm_checkpoint": str(Path(args.vlm_checkpoint)),
            "flow_checkpoint": str(Path(args.flow_checkpoint)),
            "labels": str(Path(args.labels)),
        },
        "files": {},
        "imports": {},
    }

    for key, path in report["paths"].items():
        report["files"][key] = Path(path).exists()

    minimind_root = Path(args.minimind_v_root)
    report["files"]["minimind_model_dir"] = (minimind_root / "model").is_dir()
    report["files"]["vision_model_dir"] = (minimind_root / "model" / "siglip2-base-p32-256-ve").is_dir()

    for name in ["torch", "transformers", "torchvision", "PIL", "numpy"]:
        report["imports"][name] = _try_import(name)

    try:
        import torch

        report["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        report["torch"] = {"error": str(exc)}

    try:
        sys.path.insert(0, str(minimind_root))
        loaded_model_module = sys.modules.get("model")
        if loaded_model_module is not None and not hasattr(loaded_model_module, "__path__"):
            sys.modules.pop("model", None)
        from model.model_vlm import MiniMindVLM, VLMConfig  # noqa: F401

        report["minimind_v_import"] = {"ok": True}
    except Exception as exc:
        report["minimind_v_import"] = {"ok": False, "error": str(exc)}

    print(json.dumps(report, ensure_ascii=False, indent=2))
    required_ok = all(report["files"].values()) and report["imports"].get("torch", {}).get("ok") and report["imports"].get("transformers", {}).get("ok") and report["minimind_v_import"].get("ok")
    if not required_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
