"""
tools/ai/run_minimax_backend.py
Run MiniMax Music 3 locally with the official Hugging Face Diffusers pipeline.

This is standard torch inference against MiniMaxAI/MiniMax-Music3.
Prompt/tags become the music description; lyrics stay structured lyrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-Music3"
MIN_DURATION_SECONDS = 1.0
MAX_DURATION_SECONDS = 300.0
MODEL_INDEX_NAMES = ("modular_model_index.json", "model_index.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax Music 3 locally via Diffusers")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--tags", default="")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-offload", choices=["true", "false"], default="false")
    parser.add_argument("--instrumental", choices=["true", "false"], default="false")
    return parser.parse_args()


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def compose_caption(prompt: str | None, tags: str | None) -> str:
    prompt_text = (prompt or "").strip()
    tags_text = (tags or "").strip()
    if prompt_text and tags_text:
        return f"{prompt_text}. {tags_text}"
    return prompt_text or tags_text


def compose_lyrics(lyrics: str | None, *, instrumental: bool) -> str:
    lyrics_text = (lyrics or "").strip()
    if lyrics_text:
        return lyrics_text
    if instrumental:
        return "[Instrumental]"
    return ""


def clamp_duration_seconds(duration: float) -> float:
    return min(MAX_DURATION_SECONDS, max(MIN_DURATION_SECONDS, float(duration)))


def _has_diffusers_index(root: Path) -> bool:
    return any((root / name).exists() for name in MODEL_INDEX_NAMES)


def resolve_minimax_model_dir(model_dir: str | Path | None) -> Path | None:
    if not model_dir:
        return None
    root = Path(model_dir)
    if _has_diffusers_index(root):
        return root.resolve()
    return None


def require_minimax_model_dir(model_dir: str | Path) -> Path:
    resolved = resolve_minimax_model_dir(model_dir)
    if resolved is None:
        raise FileNotFoundError(
            f"MiniMax Music 3 model dir is missing modular_model_index.json: {model_dir}. "
            f"Download the official Hugging Face repo {DEFAULT_MODEL_ID}."
        )
    return resolved


def remap_local_component_paths(model_dir: str | Path) -> Path:
    """Point ModularPipeline components at the local snapshot instead of the Hub id."""
    root = Path(model_dir).resolve()
    index_path = next((root / name for name in MODEL_INDEX_NAMES if (root / name).exists()), None)
    if index_path is None:
        return root
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    changed = False
    for value in payload.values():
        if not (isinstance(value, list) and len(value) >= 3 and isinstance(value[2], dict)):
            continue
        current = str(value[2].get("pretrained_model_name_or_path") or "")
        if current in {"", DEFAULT_MODEL_ID} or not Path(current).exists():
            value[2]["pretrained_model_name_or_path"] = str(root)
            changed = True
    if changed:
        index_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return root


def _write_audio(audio, output_path: Path, sample_rate: int) -> Path:
    import numpy as np
    import soundfile as sf

    array = audio.detach().float().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 2 and array.shape[0] <= 8:
        array = array.T
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), array, int(sample_rate))
    return output_path


def generate_minimax_audio(
    *,
    output_path: str | Path,
    prompt: str,
    tags: str = "",
    lyrics: str = "",
    model_dir: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    duration: float = 60.0,
    seed: int = 0,
    cpu_offload: bool = False,
    instrumental: bool = False,
) -> dict[str, object]:
    import torch
    from diffusers import ModularPipeline

    caption = compose_caption(prompt, tags)
    if not caption:
        raise ValueError("MiniMax Music 3 needs a prompt or tags for the music description.")
    lyrics_text = compose_lyrics(lyrics, instrumental=instrumental)
    duration_seconds = clamp_duration_seconds(duration)
    output = Path(output_path).resolve()
    pretrained = str(require_minimax_model_dir(model_dir)) if model_dir else model_id
    if model_dir:
        pretrained = str(remap_local_component_paths(pretrained))
    if not torch.cuda.is_available():
        raise RuntimeError("MiniMax Music 3 inference requires CUDA.")

    load_kwargs: dict[str, object] = {}
    if cpu_offload:
        from diffusers import ComponentsManager

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device="cuda")
        load_kwargs["components_manager"] = manager

    pipe = ModularPipeline.from_pretrained(pretrained, local_files_only=bool(model_dir), **load_kwargs)
    pipe.load_components(dtype=torch.bfloat16)
    if not cpu_offload:
        pipe.to("cuda")
    elif hasattr(pipe, "language_model"):
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            pipe.language_model,
            onload_device=torch.device("cuda"),
            offload_type="leaf_level",
            use_stream=True,
        )

    generator = torch.Generator("cuda").manual_seed(int(seed))
    result = pipe(
        prompt=caption,
        lyrics=lyrics_text,
        audio_duration=duration_seconds,
        generator=generator,
        output="audios",
    )
    audio = result[0] if isinstance(result, (list, tuple)) else result
    sample_rate = int(getattr(pipe, "sampling_rate", 32000) or 32000)
    _write_audio(audio, output, sample_rate)
    return {
        "ok": True,
        "output": str(output),
        "caption": caption,
        "lyrics": lyrics_text,
        "duration_seconds": duration_seconds,
        "generated_seconds": duration_seconds,
        "seed": int(seed),
        "cpu_offload": cpu_offload,
        "model": pretrained,
        "sample_rate": sample_rate,
        "backend": "diffusers",
    }


def main() -> None:
    args = parse_args()
    try:
        result = generate_minimax_audio(
            output_path=args.output,
            prompt=args.prompt,
            tags=args.tags,
            lyrics=args.lyrics,
            model_dir=args.model_dir or None,
            model_id=args.model_id,
            duration=args.duration,
            seed=args.seed if args.seed is not None else 0,
            cpu_offload=_as_bool(args.cpu_offload),
            instrumental=_as_bool(args.instrumental),
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
