"""MiniMax Music 3 smoke: no lyrics, instrumental instructions only."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_minimax_backend import generate_minimax_audio


PROMPT = (
    "Instrumental only. No vocals, no choir, no hummed melody. "
    "Genre: cinematic ambient folk. BPM: 84. Key: D minor. "
    "Arrangement: fingerpicked acoustic guitar, low cello pads, "
    "soft analog piano, distant brushed cymbals. "
    "Mood: late-night, spacious, melancholic, no lyrics."
)

OUT = Path(r"D:\AI\Models\music\comparison\minimax-smoke-instrumental.wav")
MODEL = Path(r"D:\AI\Models\music\comparison\minimax-music-3")
LOG = Path(r"D:\AI\Models\music\comparison\minimax-smoke-instrumental.log")


def _log(message: str) -> None:
    line = message.rstrip() + "\n"
    print(line, end="", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)


def main() -> None:
    _log(f"starting MiniMax instrumental smoke -> {OUT}")
    try:
        result = generate_minimax_audio(
            output_path=OUT,
            prompt=PROMPT,
            lyrics="",
            model_dir=MODEL,
            duration=16.0,
            seed=11,
            cpu_offload=False,
            instrumental=True,
        )
    except Exception as exc:
        _log(f"FAILED {type(exc).__name__}: {exc}")
        raise
    _log(str(result))


if __name__ == "__main__":
    main()
