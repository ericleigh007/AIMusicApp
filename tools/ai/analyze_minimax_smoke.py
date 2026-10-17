"""Two independent analyzers for a MiniMax smoke WAV.

1) SignalAnalyzer  — energy, stereo, spectrum shape, silence
2) MusicAnalyzer   — tempo-like periodicity, harmonic peakiness, onset density

Neither is a genre classifier. Together they answer: is this silence, noise,
a static tone, or something that looks like music.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_wav(path: Path) -> tuple["object", int]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), always_2d=True)
    audio = np.asarray(audio, dtype=np.float64)
    return audio, int(sample_rate)


def _mono(audio) -> "object":
    import numpy as np

    return np.mean(audio, axis=1)


def analyze_signal(audio, sample_rate: int) -> dict[str, object]:
    import numpy as np

    mono = _mono(audio)
    duration = float(mono.shape[0] / sample_rate)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if mono.size else 0.0
    silent = float(np.mean(np.abs(mono) < 1e-3)) if mono.size else 1.0
    dc = float(np.mean(mono)) if mono.size else 0.0
    stereo_corr = None
    if audio.shape[1] >= 2:
        left, right = audio[:, 0], audio[:, 1]
        if np.std(left) > 1e-8 and np.std(right) > 1e-8:
            stereo_corr = float(np.corrcoef(left, right)[0, 1])

    window = np.hanning(min(mono.size, sample_rate))
    if window.size < 256:
        centroid_hz = 0.0
        flatness = 1.0
        high_band = 0.0
    else:
        snippet = mono[: window.size] * window
        spec = np.abs(np.fft.rfft(snippet)) + 1e-12
        freqs = np.fft.rfftfreq(window.size, 1.0 / sample_rate)
        centroid_hz = float(np.sum(freqs * spec) / np.sum(spec))
        log_mean = float(np.exp(np.mean(np.log(spec))))
        flatness = float(log_mean / np.mean(spec))
        high_band = float(np.sum(spec[freqs > 8000]) / np.sum(spec))

    music_like = (
        duration >= 4.0
        and peak >= 0.02
        and rms >= 0.005
        and silent < 0.65
        and 80.0 <= centroid_hz <= 6000.0
        and flatness < 0.55
        and high_band < 0.45
        and abs(dc) < 0.05
    )
    return {
        "analyzer": "signal",
        "duration_seconds": round(duration, 3),
        "channels": int(audio.shape[1]),
        "sample_rate": sample_rate,
        "peak": round(peak, 5),
        "rms": round(rms, 5),
        "silence_ratio": round(silent, 4),
        "dc_offset": round(dc, 5),
        "stereo_correlation": None if stereo_corr is None else round(stereo_corr, 4),
        "spectral_centroid_hz": round(centroid_hz, 1),
        "spectral_flatness": round(flatness, 4),
        "high_band_ratio": round(high_band, 4),
        "music_like": bool(music_like),
    }


def analyze_music(audio, sample_rate: int) -> dict[str, object]:
    import numpy as np

    mono = _mono(audio)
    if mono.size < sample_rate:
        return {
            "analyzer": "music",
            "music_like": False,
            "error": "audio shorter than 1 second",
        }

    hop = max(1, sample_rate // 100)
    frame = max(hop * 2, sample_rate // 50)
    envelope = []
    for start in range(0, mono.size - frame, hop):
        chunk = mono[start : start + frame]
        envelope.append(float(np.sqrt(np.mean(chunk * chunk))))
    env = np.asarray(envelope, dtype=np.float64)
    env = env - np.mean(env)
    bpm = None
    tempo_strength = 0.0
    if env.size > 32 and float(np.max(np.abs(env))) > 1e-8:
        corr = np.correlate(env, env, mode="full")[env.size - 1 :]
        min_lag = int(round(60.0 / 180.0 * (sample_rate / hop)))
        max_lag = int(round(60.0 / 50.0 * (sample_rate / hop)))
        max_lag = min(max_lag, corr.size - 1)
        if max_lag > min_lag + 2:
            region = corr[min_lag:max_lag]
            lag = int(np.argmax(region)) + min_lag
            bpm = float(60.0 * (sample_rate / hop) / lag)
            # Autocorr often locks onto the half-beat (double-time).
            if bpm > 160.0:
                bpm = bpm / 2.0
            denom = float(corr[0]) if corr[0] else 1.0
            tempo_strength = float(region.max() / denom)

    hop_spec = sample_rate // 20
    n_fft = 2048
    window = np.hanning(n_fft)
    peakiness = []
    onsets = 0
    prev = None
    for start in range(0, mono.size - n_fft, hop_spec):
        spec = np.abs(np.fft.rfft(mono[start : start + n_fft] * window)) + 1e-12
        peakiness.append(float(np.max(spec) / np.mean(spec)))
        if prev is not None:
            flux = float(np.mean(np.clip(spec - prev, 0, None)))
            if flux > 0.02:
                onsets += 1
        prev = spec
    harmonic_peakiness = float(np.median(peakiness)) if peakiness else 0.0
    duration = float(mono.shape[0] / sample_rate)
    onset_per_sec = onsets / duration if duration else 0.0

    music_like = (
        bpm is not None
        and 50.0 <= bpm <= 180.0
        and tempo_strength >= 0.08
        and harmonic_peakiness >= 8.0
        and 0.2 <= onset_per_sec <= 24.0
    )
    return {
        "analyzer": "music",
        "estimated_bpm": None if bpm is None else round(bpm, 1),
        "tempo_strength": round(tempo_strength, 4),
        "harmonic_peakiness": round(harmonic_peakiness, 2),
        "onset_density_per_sec": round(onset_per_sec, 3),
        "music_like": bool(music_like),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a MiniMax smoke WAV")
    parser.add_argument("wav")
    args = parser.parse_args()
    path = Path(args.wav)
    if not path.exists():
        print(json.dumps({"ok": False, "error": f"missing {path}"}))
        return 2
    audio, sample_rate = _load_wav(path)
    signal = analyze_signal(audio, sample_rate)
    music = analyze_music(audio, sample_rate)
    payload = {
        "ok": bool(signal.get("music_like") and music.get("music_like")),
        "path": str(path),
        "bytes": path.stat().st_size,
        "signal": signal,
        "music": music,
        "verdict": (
            "music-like"
            if signal.get("music_like") and music.get("music_like")
            else "not convincingly music"
        ),
    }
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
