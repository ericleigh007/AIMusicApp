"""
tools/ai/run_yue2_backend.py
Local YuE2 runner for the comparison app.

Uses the official yue2.YuE2Pipeline API (m-a-p/YuE2-3B + m-a-p/YuE2-Vae).
Song length follows the symbolic plan. Duration is not a pipeline argument.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


YUE2_MODEL_ID = "m-a-p/YuE2-3B"
YUE2_VAE_ID = "m-a-p/YuE2-Vae"
YUE2_COT_CHOICES = ("full", "melody", "off")
YUE2_BACKEND_CHOICES = ("torch", "torch-eager", "vllm")


def compose_style(prompt: str | None, tags: str | None) -> str:
    prompt_text = (prompt or "").strip()
    tags_text = (tags or "").strip()
    if prompt_text and tags_text:
        return f"{prompt_text}. {tags_text}"
    return prompt_text or tags_text


def compose_lyrics(lyrics: str | None) -> tuple[str, bool]:
    text = (lyrics or "").strip()
    if text:
        return text, False
    return "[Instrumental]", True


def _has_weight_file(directory: Path) -> bool:
    return (directory / "model.safetensors").is_file() or (directory / "model.safetensors.index.json").is_file()


def resolve_yue2_dirs(model_dir: str | Path, vae_dir: str | Path | None = None) -> tuple[Path, Path]:
    model_path = Path(model_dir).expanduser()
    vae_path = Path(vae_dir).expanduser() if vae_dir else None
    if (model_path / "YuE2-3B").is_dir() and not (model_path / "qwen.tiktoken").is_file():
        if vae_path is None or not str(vae_path).strip():
            vae_path = model_path / "YuE2-Vae"
        model_path = model_path / "YuE2-3B"
    if vae_path is None or not str(vae_path).strip():
        sibling = model_path.parent / "YuE2-Vae"
        vae_path = sibling
    return model_path, vae_path


def require_yue2_model_dir(model_dir: str | Path) -> Path:
    resolved, _vae = resolve_yue2_dirs(model_dir, None)
    if not resolved.is_dir():
        raise FileNotFoundError(f"YuE2 model directory does not exist: {resolved}")
    if not (resolved / "qwen.tiktoken").is_file() or not _has_weight_file(resolved):
        raise FileNotFoundError(
            f"YuE2 model directory is not an m-a-p/YuE2-3B snapshot (need qwen.tiktoken and model.safetensors): {resolved}"
        )
    return resolved.resolve()


def require_yue2_vae_dir(vae_dir: str | Path) -> Path:
    resolved = Path(vae_dir).expanduser()
    if not resolved.is_dir():
        raise FileNotFoundError(f"YuE2 VAE directory does not exist: {resolved}")
    if not (resolved / "config.json").is_file() or not _has_weight_file(resolved):
        raise FileNotFoundError(
            f"YuE2 VAE directory is not an m-a-p/YuE2-Vae snapshot (need config.json and model.safetensors): {resolved}"
        )
    return resolved.resolve()


def generate_yue2_audio(
    *,
    prompt: str,
    tags: str,
    lyrics: str,
    output_path: str | Path,
    model_dir: str | Path,
    vae_dir: str | Path,
    cot: str = "full",
    seed: int = 831001,
    cfg_scale: float | None = None,
    abc_text: str | None = None,
    device: str = "cuda",
    runtime_backend: str = "torch",
    memory_budget_gib: float = 24.0,
    offload_ar: bool = False,
) -> dict[str, object]:
    if cot not in YUE2_COT_CHOICES:
        raise ValueError(f"cot must be one of {', '.join(YUE2_COT_CHOICES)}")
    if runtime_backend not in YUE2_BACKEND_CHOICES:
        raise ValueError(f"runtime backend must be one of {', '.join(YUE2_BACKEND_CHOICES)}")
    style = compose_style(prompt, tags)
    if not style:
        raise ValueError("YuE2 needs a style prompt. Fill Prompt/Chat and/or Tags.")
    lyric_text, instrumental = compose_lyrics(lyrics)
    abc = (abc_text or "").strip() or None
    if abc and cot == "off":
        raise ValueError("An ABC score requires cot=full or cot=melody.")

    model_path = require_yue2_model_dir(model_dir)
    vae_path = require_yue2_vae_dir(vae_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output.parent / "yue2_artifacts"

    from yue2 import YuE2Pipeline

    request_kwargs: dict[str, object] = {
        "style": style,
        "lyrics": lyric_text,
        "cot": cot,
        "seed": int(seed),
        "id": "aimusicapp",
    }
    if abc:
        request_kwargs["abc"] = abc
    if cfg_scale is not None:
        request_kwargs["cfg_scale"] = float(cfg_scale)

    with YuE2Pipeline.from_pretrained(
        str(model_path),
        vae=str(vae_path),
        device=device,
        memory_budget_gib=float(memory_budget_gib),
        backend=runtime_backend,
        offload_ar=bool(offload_ar),
        local_files_only=True,
        progress=False,
    ) as pipe:
        song = pipe(**request_kwargs)
        song.save(output)
        song.save_artifacts(artifacts_dir)

    audio_seconds = float(len(song.audio) / song.sample_rate) if song.sample_rate else 0.0
    return {
        "backend": "yue2",
        "model_id": YUE2_MODEL_ID,
        "vae_id": YUE2_VAE_ID,
        "model_dir": str(model_path),
        "vae_dir": str(vae_path),
        "style": style,
        "lyrics": lyric_text,
        "instrumental": instrumental,
        "cot": cot,
        "seed": int(seed),
        "cfg_scale": cfg_scale,
        "runtime_backend": runtime_backend,
        "memory_budget_gib": float(memory_budget_gib),
        "offload_ar": bool(offload_ar),
        "sample_rate": int(song.sample_rate),
        "audio_seconds": audio_seconds,
        "truncated": song.truncated,
        "artifacts_dir": str(artifacts_dir),
        "output": str(output),
    }


def _parse_optional_float(raw: str | None) -> float | None:
    text = (raw or "").strip()
    if not text:
        return None
    return float(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one YuE2 song for AIMusicApp")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--cot", default="full", choices=list(YUE2_COT_CHOICES))
    parser.add_argument("--seed", type=int, default=831001)
    parser.add_argument("--cfg-scale", default="")
    parser.add_argument("--abc-file", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runtime-backend", default="torch", choices=list(YUE2_BACKEND_CHOICES))
    parser.add_argument("--memory-budget-gib", type=float, default=24.0)
    parser.add_argument("--offload-ar", default="false")
    args = parser.parse_args()

    abc_text = ""
    if args.abc_file.strip():
        abc_path = Path(args.abc_file)
        if not abc_path.is_file():
            raise SystemExit(f"ABC file does not exist: {abc_path}")
        abc_text = abc_path.read_text(encoding="utf-8")

    try:
        payload = generate_yue2_audio(
            prompt=args.prompt,
            tags=args.tags,
            lyrics=args.lyrics,
            output_path=args.output,
            model_dir=args.model_dir,
            vae_dir=args.vae_dir,
            cot=args.cot,
            seed=args.seed,
            cfg_scale=_parse_optional_float(args.cfg_scale),
            abc_text=abc_text,
            device=args.device,
            runtime_backend=args.runtime_backend,
            memory_budget_gib=args.memory_budget_gib,
            offload_ar=args.offload_ar.strip().lower() in {"1", "true", "yes", "on"},
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
