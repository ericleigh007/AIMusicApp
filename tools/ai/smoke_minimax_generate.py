"""One-shot MiniMax Music 3 smoke generate (official Diffusers, no Comfy)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_minimax_backend import generate_minimax_audio


PROMPT = (
    "Genre: acoustic folk. BPM: 92. Key: G major. Warm and intimate. "
    "Vocals: close male lead, gentle and clear, light harmony on the last line. "
    "Arrangement: fingerpicked acoustic guitar and soft upright bass; "
    "brushed snare enters on the chorus."
)

LYRICS = """[verse]
Coffee cooling on the windowsill
Rain keeps time against the glass
I count the quiet I can still
[chorus]
Hold the morning, let it last
Hold the morning, let it last
"""

OUT = Path(r"D:\AI\Models\music\comparison\minimax-smoke.wav")
MODEL = Path(r"D:\AI\Models\music\comparison\minimax-music-3")


LOG = Path(r"D:\AI\Models\music\comparison\minimax-smoke.log")


def _log(message: str) -> None:
    line = message.rstrip() + "\n"
    print(line, end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def main() -> None:
    _log(f"starting MiniMax smoke -> {OUT}")
    _log(f"model={MODEL}")
    try:
        result = generate_minimax_audio(
            output_path=OUT,
            prompt=PROMPT,
            lyrics=LYRICS,
            model_dir=MODEL,
            duration=16.0,
            seed=7,
            cpu_offload=False,
            instrumental=False,
        )
    except Exception as exc:
        _log(f"FAILED {type(exc).__name__}: {exc}")
        raise
    _log(str(result))


if __name__ == "__main__":
    main()
