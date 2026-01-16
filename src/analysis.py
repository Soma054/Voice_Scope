from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import librosa


METRIC_KEYS = ["stability", "clarity", "resonance", "brightness", "voiced"]


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _to_score01(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return _clamp01((x - lo) / (hi - lo))


def _to_score100(x: float, lo: float, hi: float) -> int:
    return int(round(_to_score01(x, lo, hi) * 100))


def _safe_db(x: np.ndarray, ref: float = 1.0) -> np.ndarray:
    x = np.asarray(x)
    return 20.0 * np.log10(np.maximum(1e-10, x / ref))


def hz_to_note(hz: float) -> str:
    if not np.isfinite(hz) or hz <= 0:
        return "-"
    try:
        return librosa.hz_to_note(float(hz))
    except Exception:
        return "-"


@dataclass(frozen=True)
class MetricResult:
    """
    score: 0-100 (higher is better)
    raw:   raw feature values (debug/advanced)
    """
    score: Dict[str, int]
    raw: Dict[str, float]


def metric_labels_jp() -> Dict[str, str]:
    return {
        "stability": "安定性",
        "clarity": "クリアさ",
        "resonance": "響き・抜け",
        "brightness": "明るさ",
        "voiced": "発声率",
    }


def compute_voice_metrics(y: np.ndarray, sr: int) -> MetricResult:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        y = np.mean(y, axis=1)

    # Trim silence (helps pitch extraction)
    y_trim, _ = librosa.effects.trim(y, top_db=35)
    y_use = y_trim if len(y_trim) > sr * 0.2 else y

    # --- Brightness: spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=y_use, sr=sr)[0]
    centroid_hz = float(np.median(centroid))
    brightness_score = _to_score100(centroid_hz, lo=800.0, hi=3500.0)

    # --- Resonance / cut-through proxy: spectral rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y_use, sr=sr, roll_percent=0.85)[0]
    rolloff_hz = float(np.median(rolloff))
    resonance_score = _to_score100(rolloff_hz, lo=2000.0, hi=8000.0)

    # --- Clarity proxy: inverted spectral flatness ---
    flatness = librosa.feature.spectral_flatness(y=y_use)[0]
    flatness_med = float(np.median(flatness))
    clarity_score = 100 - _to_score100(flatness_med, lo=0.02, hi=0.25)

    # --- Pitch-based features via pyin ---
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y_use,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )

    voiced_mask = np.isfinite(f0)
    f0_voiced = f0[voiced_mask]

    # Voiced ratio: percent of frames that have valid f0
    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0
    # Normalize: 0.35 -> 0, 0.90 -> 100 (歌/音源でだいたいこの辺に収まりやすい)
    voiced_score = _to_score100(voiced_ratio, lo=0.35, hi=0.90)

    # Stability: f0 CV (lower is better)
    if len(f0_voiced) < 10:
        stability_score = 0
        f0_mean = float("nan")
        f0_cv = float("nan")
        f0_p05 = float("nan")
        f0_p95 = float("nan")
    else:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        f0_cv = float(f0_std / max(1e-6, f0_mean))

        # Normalize: CV 0.03 (very stable) -> 100, CV 0.18 (unstable) -> 0
        stability_score = 100 - _to_score100(f0_cv, lo=0.03, hi=0.18)

        # Range info (not a radar axis, but great for display)
        f0_p05 = float(np.percentile(f0_voiced, 5))
        f0_p95 = float(np.percentile(f0_voiced, 95))

    score = {
        "stability": int(np.clip(stability_score, 0, 100)),
        "clarity": int(np.clip(clarity_score, 0, 100)),
        "resonance": int(np.clip(resonance_score, 0, 100)),
        "brightness": int(np.clip(brightness_score, 0, 100)),
        "voiced": int(np.clip(voiced_score, 0, 100)),
    }

    # Some extra raw values to support “盛り” UI
    rms = librosa.feature.rms(y=y_use)[0]
    rms_db_med = float(np.median(_safe_db(rms, ref=1.0)))
    rms_db_iqr = float(np.percentile(_safe_db(rms, ref=1.0), 75) - np.percentile(_safe_db(rms, ref=1.0), 25))

    raw = {
        "centroid_hz_median": centroid_hz,
        "rolloff85_hz_median": rolloff_hz,
        "flatness_median": flatness_med,
        "voiced_ratio": voiced_ratio,
        "f0_mean_hz": f0_mean,
        "f0_cv": f0_cv,
        "f0_p05_hz": f0_p05,
        "f0_p95_hz": f0_p95,
        "f0_p05_note": hz_to_note(f0_p05),
        "f0_p95_note": hz_to_note(f0_p95),
        "rms_db_median": rms_db_med,
        "rms_db_iqr": rms_db_iqr,
    }

    return MetricResult(score=score, raw=raw)