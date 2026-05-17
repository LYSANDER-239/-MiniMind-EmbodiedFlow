import json
import os
import re
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image


CANONICAL_DEFAULTS = {
    "action_type": "move",
    "avoid_color": "orange",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARSE_FIELDS = ["target_color", "target_shape", "goal_color", "action_type", "avoid_color"]

COLOR_SYNONYMS = {
    "red": ["red", "红色", "红"],
    "yellow": ["yellow", "黄色", "黄"],
    "green": ["green", "绿色", "绿"],
    "purple": ["purple", "紫色", "紫"],
    "blue": ["blue", "蓝色", "蓝"],
    "orange": ["orange", "橙色", "橙"],
}

SHAPE_SYNONYMS = {
    "square": ["square", "方形", "正方形", "方块"],
    "circle": ["circle", "圆形", "圆球", "圆"],
    "triangle": ["triangle", "三角形", "三角"],
}


def is_parse_valid(parsed: dict | None) -> bool:
    if not isinstance(parsed, dict):
        return False
    if parsed.get("error"):
        return False
    return all(parsed.get(field) is not None for field in PARSE_FIELDS)


def configure_temp_dir():
    temp_dir = PROJECT_ROOT / "tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(temp_dir))
    os.environ.setdefault("TEMP", str(temp_dir))
    os.environ.setdefault("TMP", str(temp_dir))
    tempfile.tempdir = str(temp_dir)

COLOR_VALUE_MAP = {
    "red": "red",
    "yellow": "yellow",
    "green": "green",
    "purple": "purple",
    "blue": "blue",
    "orange": "orange",
    "红": "red",
    "红色": "red",
    "黄": "yellow",
    "黄色": "yellow",
    "绿": "green",
    "绿色": "green",
    "紫": "purple",
    "紫色": "purple",
    "蓝": "blue",
    "蓝色": "blue",
    "橙": "orange",
    "橙色": "orange",
}
for canonical, words in COLOR_SYNONYMS.items():
    for word in words:
        COLOR_VALUE_MAP[word] = canonical

SHAPE_VALUE_MAP = {
    "square": "square",
    "circle": "circle",
    "triangle": "triangle",
    "方形": "square",
    "正方形": "square",
    "方块": "square",
    "圆形": "circle",
    "圆球": "circle",
    "圆": "circle",
    "三角形": "triangle",
    "三角": "triangle",
}
for canonical, words in SHAPE_SYNONYMS.items():
    for word in words:
        SHAPE_VALUE_MAP[word] = canonical

ACTION_VALUE_MAP = {
    "move": "move",
    "移动": "move",
    "移动到": "move",
    "放到": "move",
    "放进": "move",
    "送到": "move",
    "移至": "move",
    "前往": "move",
    "到达": "move",
    "搬运": "move",
    "控制": "move",
}

FIELD_ALIASES = {
    "target_color": ["target_color", "targetColor", "object_color", "color", "目标颜色", "物体颜色", "目标物体颜色"],
    "target_shape": ["target_shape", "targetShape", "object_shape", "shape", "目标形状", "物体形状", "目标物体形状"],
    "goal_color": ["goal_color", "goalColor", "region_color", "target_region_color", "目标区域颜色", "区域颜色"],
    "action_type": ["action_type", "actionType", "action", "动作", "动作类型"],
    "avoid_color": ["avoid_color", "avoidColor", "obstacle_color", "avoid", "避障颜色", "障碍物颜色"],
}


def device_from_arg(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def build_prompt(instruction: str, prompt_mode: str = "zero_shot", image_placeholder: str = "<image>") -> str:
    schema = (
        'JSON 字段必须为：\n'
        '{\n'
        '  "target_color": "...",\n'
        '  "target_shape": "...",\n'
        '  "goal_color": "...",\n'
        '  "action_type": "move",\n'
        '  "avoid_color": "orange"\n'
        '}\n\n'
        '字段取值范围：\n'
        'target_color: red, yellow, green, purple\n'
        'target_shape: square, circle, triangle\n'
        'goal_color: blue, green, yellow\n'
        'action_type: move\n'
        'avoid_color: orange'
    )
    if prompt_mode == "few_shot":
        examples = (
            "示例：\n"
            '指令：把黄色圆形移动到蓝色区域，并绕开橙色障碍物。\n'
            '输出：{"target_color":"yellow","target_shape":"circle","goal_color":"blue","action_type":"move","avoid_color":"orange"}\n\n'
            '指令：请将红色方块放进绿色目标区域。\n'
            '输出：{"target_color":"red","target_shape":"square","goal_color":"green","action_type":"move","avoid_color":"orange"}\n\n'
            '指令：将紫色三角形移至黄色区域，不要碰到橙色障碍物。\n'
            '输出：{"target_color":"purple","target_shape":"triangle","goal_color":"yellow","action_type":"move","avoid_color":"orange"}\n\n'
        )
    else:
        examples = ""

    return (
        f"{image_placeholder}\n"
        "请根据图像和指令，输出任务解析 JSON。\n\n"
        f"{examples}"
        f"指令：{instruction}\n\n"
        "只输出 JSON，不要解释，不要 Markdown，不要代码块。\n\n"
        f"{schema}"
    )


def _strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _find_json_object(text: str) -> str | None:
    text = _strip_code_fence(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:idx + 1]
    return None


def extract_json(raw_output: str) -> tuple[dict | None, str | None]:
    json_text = _find_json_object(raw_output)
    if not json_text:
        return None, "json_not_found"
    try:
        return json.loads(json_text), None
    except json.JSONDecodeError:
        repaired = json_text.replace("'", '"')
        repaired = re.sub(r",\s*}", "}", repaired)
        try:
            return json.loads(repaired), None
        except json.JSONDecodeError as exc:
            return None, f"json_decode_failed:{exc.msg}"


def _pick_field(parsed: dict, canonical_field: str):
    for alias in FIELD_ALIASES[canonical_field]:
        if alias in parsed:
            return parsed.get(alias)
    return None


def _normalize_value(value, mapping: dict[str, str]):
    if value is None:
        return None
    value = str(value).strip().strip('"').strip("'")
    return mapping.get(value, mapping.get(value.lower(), value.lower()))


def normalize_parse(parsed: dict | None) -> tuple[dict, str | None]:
    normalized = {
        "target_color": None,
        "target_shape": None,
        "goal_color": None,
        "action_type": CANONICAL_DEFAULTS["action_type"],
        "avoid_color": CANONICAL_DEFAULTS["avoid_color"],
    }
    if not isinstance(parsed, dict):
        normalized["error"] = "invalid_json_object"
        return normalized, normalized["error"]

    normalized["target_color"] = _normalize_value(_pick_field(parsed, "target_color"), COLOR_VALUE_MAP)
    normalized["target_shape"] = _normalize_value(_pick_field(parsed, "target_shape"), SHAPE_VALUE_MAP)
    normalized["goal_color"] = _normalize_value(_pick_field(parsed, "goal_color"), COLOR_VALUE_MAP)
    action = _normalize_value(_pick_field(parsed, "action_type"), ACTION_VALUE_MAP)
    avoid = _normalize_value(_pick_field(parsed, "avoid_color"), COLOR_VALUE_MAP)
    if action:
        normalized["action_type"] = action
    if avoid:
        normalized["avoid_color"] = avoid

    allowed = {
        "target_color": {"red", "yellow", "green", "purple"},
        "target_shape": {"square", "circle", "triangle"},
        "goal_color": {"blue", "green", "yellow"},
        "action_type": {"move"},
        "avoid_color": {"orange"},
    }
    invalid = [
        field for field, allowed_values in allowed.items()
        if normalized.get(field) not in allowed_values
    ]
    missing = [field for field in PARSE_FIELDS if normalized.get(field) is None]
    if missing:
        error = "missing_" + "_".join(missing)
    elif invalid:
        error = "invalid_" + "_".join(invalid)
    else:
        error = None
    normalized["error"] = error
    return normalized, error


class MiniMindVParser:
    def __init__(
        self,
        minimind_v_root: str,
        model_path: str,
        device: str = "auto",
        prompt_mode: str = "zero_shot",
        image_placeholder: str = "<image>",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: int = 0,
    ):
        self.minimind_v_root = Path(minimind_v_root).expanduser().resolve()
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device_from_arg(device)
        self.prompt_mode = prompt_mode
        self.image_placeholder = image_placeholder
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe

        if not self.minimind_v_root.is_dir():
            raise RuntimeError(f"model_load_failed: MiniMind-V root not found: {self.minimind_v_root}")
        if not self.model_path.is_file():
            raise RuntimeError(f"missing_checkpoint: {self.model_path}")
        if self.device.type == "cpu":
            print("Warning: MiniMind-V inference on CPU may be very slow.")

        configure_temp_dir()
        sys.path.insert(0, str(self.minimind_v_root))
        try:
            from transformers import AutoTokenizer
            loaded_model_module = sys.modules.get("model")
            if loaded_model_module is not None and not hasattr(loaded_model_module, "__path__"):
                # The Flow demo has a top-level model.py. Remove that cached module
                # so MiniMind-V's official model/ package can be imported.
                sys.modules.pop("model", None)
            from model.model_vlm import MiniMindVLM, VLMConfig
        except Exception as exc:
            raise RuntimeError(f"unsupported_minimind_v_api: {exc}") from exc

        try:
            tokenizer_path = self.minimind_v_root / "model"
            vision_model_path = self.minimind_v_root / "model" / "siglip2-base-p32-256-ve"
            self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
            self.model = MiniMindVLM(
                VLMConfig(
                    hidden_size=hidden_size,
                    num_hidden_layers=num_hidden_layers,
                    use_moe=bool(use_moe),
                ),
                vision_model_path=str(vision_model_path),
            )
            if self.model.vision_encoder is None or self.model.processor is None:
                raise RuntimeError(f"vision model not found or failed to load: {vision_model_path}")
            state_dict = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict({k: v for k, v in state_dict.items() if "mask" not in k}, strict=False)
            self.model.eval().to(self.device)
            if self.device.type == "cuda":
                self.model = self.model.half()
            else:
                self.model = self.model.float()
                if self.model.vision_encoder is not None:
                    self.model.vision_encoder = self.model.vision_encoder.float()
            self.preprocess = self.model.processor
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"model_load_failed: {exc}") from exc

    def _image_to_pixel_values(self, image_path: str):
        try:
            loaded_model_module = sys.modules.get("model")
            if loaded_model_module is not None and not hasattr(loaded_model_module, "__path__"):
                sys.modules.pop("model", None)
            from model.model_vlm import MiniMindVLM

            image = Image.open(image_path).convert("RGB")
            pixel_values = MiniMindVLM.image2tensor(image, self.preprocess)
            return {k: v.to(self.device) for k, v in pixel_values.items()}
        except Exception as exc:
            raise RuntimeError(f"image_preprocess_failed: {exc}") from exc

    def _prompt_to_inputs(self, prompt: str):
        content = prompt.replace(
            self.image_placeholder,
            self.model.config.image_special_token * self.model.config.image_token_len,
        )
        messages = [{"role": "user", "content": content}]
        try:
            inputs_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                open_thinking=False,
            )
        except TypeError:
            inputs_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return self.tokenizer(inputs_text, return_tensors="pt", truncation=True).to(self.device)

    def generate(self, image_path: str, instruction: str) -> tuple[str, str]:
        prompt = build_prompt(instruction, self.prompt_mode, self.image_placeholder)
        pixel_values = self._image_to_pixel_values(image_path)
        inputs = self._prompt_to_inputs(prompt)
        do_sample = self.temperature > 0.0
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    do_sample=do_sample,
                    temperature=max(self.temperature, 1e-6) if do_sample else 1.0,
                    top_p=0.85,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pixel_values=pixel_values,
                )
            input_len = inputs["input_ids"].shape[1]
            raw_output = self.tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True).strip()
            return prompt, raw_output
        except Exception as exc:
            raise RuntimeError(f"generation_failed: {exc}") from exc

    def parse(self, image_path: str, instruction: str) -> dict:
        prompt, raw_output = self.generate(image_path, instruction)
        json_obj, json_error = extract_json(raw_output)
        pred_parse, normalize_error = normalize_parse(json_obj)
        json_valid = json_obj is not None
        parse_valid = is_parse_valid(pred_parse)
        return {
            "prompt": prompt,
            "raw_output": raw_output,
            "json_object": json_obj,
            "pred_parse": pred_parse,
            "json_valid": json_valid,
            "parse_valid": parse_valid,
            "error": json_error or normalize_error,
        }
