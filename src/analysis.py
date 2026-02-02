from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import librosa
from scipy.signal import butter, medfilt, sosfiltfilt


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

    # For pitch extraction, emphasize harmonic content (helps with accompaniment/percussive mix)
    try:
        y_harm, _ = librosa.effects.hpss(y_use)
    except Exception:
        y_harm = y_use

    # Band-limit for pitch tracking (reduce bass/drums + overly bright accompaniment)
    try:
        sos = butter(
            4,
            # Lower bound: include low male fundamentals; Upper bound: keep mostly voice range.
            [50.0, 1500.0],
            btype="bandpass",
            fs=float(sr),
            output="sos",
        )
        y_pitch = sosfiltfilt(sos, y_harm).astype(np.float32, copy=False)
    except Exception:
        y_pitch = y_harm

    # --- Brightness: spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=y_use, sr=sr)[0]
    centroid_hz = float(np.median(centroid))
    # NOTE: The previous range saturated too easily (100s everywhere). Keep 0–100 but
    # make 100 harder to reach so differences show up in typical recordings.
    brightness_score = _to_score100(centroid_hz, lo=700.0, hi=5200.0)

    # --- Resonance / cut-through proxy: spectral rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y_use, sr=sr, roll_percent=0.85)[0]
    rolloff_hz = float(np.median(rolloff))
    resonance_score = _to_score100(rolloff_hz, lo=1800.0, hi=12000.0)

    # --- Clarity proxy: inverted spectral flatness ---
    flatness = librosa.feature.spectral_flatness(y=y_use)[0]
    flatness_med = float(np.median(flatness))
    # Wider range to avoid frequent 100s.
    clarity_score = 100 - _to_score100(flatness_med, lo=0.01, hi=0.35)

    # --- Pitch-based features ---
    # Singing + accompaniment is hard. We try a couple of pitch-tracking configs
    # and pick the one that yields the most usable voiced frames.
    def _pyin_try(y_in: np.ndarray, *, frame_length: int, hop_length: int):
        f0, _, _ = librosa.pyin(
            y_in,
            # Expand range a bit (low male + very high singing)
            fmin=librosa.note_to_hz("A1"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        voiced_mask = np.isfinite(f0)
        f0_voiced_local = f0[voiced_mask]
        voiced_ratio_local = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0
        return f0, voiced_mask, f0_voiced_local, voiced_ratio_local

    def _residual_dispersion_cents(f0_voiced_local: np.ndarray, *, hop_length: int) -> tuple[str, int, float]:
        """Return (note, used_frames, robust_std_cents).

        note:
          - ok
          - too_few_voiced_frames
          - too_few_inlier_frames
        """
        if len(f0_voiced_local) < 10:
            return "too_few_voiced_frames", 0, float("inf")

        f0_log2_local = np.log2(np.maximum(1e-6, f0_voiced_local))

        # target ~0.25s smoothing window (keeps melody, reduces vibrato)
        win = int(round(0.25 * float(sr) / float(hop_length)))
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        win = int(min(win, max(3, (len(f0_log2_local) // 2) * 2 + 1)))
        f0_smooth_local = medfilt(f0_log2_local, kernel_size=win) if len(f0_log2_local) >= 5 else f0_log2_local

        residual_cents_local = 1200.0 * (f0_log2_local - f0_smooth_local)
        inlier_mask_local = np.abs(residual_cents_local) <= 300.0
        residual_inliers_local = residual_cents_local[inlier_mask_local]
        used_frames_local = int(len(residual_inliers_local))
        if used_frames_local < 10:
            return "too_few_inlier_frames", used_frames_local, float("inf")

        med_local = float(np.median(residual_inliers_local))
        mad_local = float(np.median(np.abs(residual_inliers_local - med_local)))
        robust_std_local = float(1.4826 * mad_local)
        return "ok", used_frames_local, robust_std_local

    # Passes: (signal, frame_length, hop_length)
    candidates = [
        (y_pitch, 4096, 256),
        (y_harm, 4096, 256),
        (y_pitch, 8192, 512),
        (y_use, 4096, 256),
    ]

    best = None
    for sig, fl, hl in candidates:
        try:
            cand = _pyin_try(sig, frame_length=fl, hop_length=hl)
        except Exception:
            continue

        f0_c, mask_c, f0v_c, vr_c = cand
        note_c, used_frames_c, robust_std_c = _residual_dispersion_cents(f0v_c, hop_length=hl)

        # Primary: pick a candidate that yields stable inlier residuals.
        # Secondary: prefer more usable frames and higher voiced coverage.
        # Sorting tuple is ascending, so use negatives where “higher is better”.
        has_ok = 1 if note_c == "ok" else 0
        sort_key = (
            has_ok,
            used_frames_c,
            -float(robust_std_c),
            float(vr_c),
            float(len(f0v_c)),
        )
        if best is None or sort_key > best[0]:
            best = (sort_key, cand)

    if best is None:
        f0 = np.array([], dtype=float)
        voiced_mask = np.array([], dtype=bool)
        f0_voiced = np.array([], dtype=float)
        voiced_ratio = 0.0
    else:
        f0, voiced_mask, f0_voiced, voiced_ratio = best[1]

    # voiced_ratio is computed from selected track
    # Wider range to keep headroom in clean recordings.
    voiced_score = _to_score100(voiced_ratio, lo=0.25, hi=0.95)

    # Stability:
    # - Measure local instability (jitter-like) around a smoothed pitch contour.
    # - Robustify against occasional octave errors / accompaniment-induced mis-tracking.
    if len(f0_voiced) < 10:
        stability_score = 0
        f0_mean = float("nan")
        f0_cv = float("nan")
        f0_p05 = float("nan")
        f0_p95 = float("nan")
        f0_residual_std_cents = float("nan")
        f0_residual_mad_cents = float("nan")
        f0_voiced_frames = int(len(f0_voiced))
        f0_used_frames = 0
        stability_note = "too_few_voiced_frames"
    else:
        f0_mean = float(np.mean(f0_voiced))
        f0_std = float(np.std(f0_voiced))
        f0_cv = float(f0_std / max(1e-6, f0_mean))

        # Smooth pitch contour in log2 domain (captures melody), then measure residual in cents.
        f0_log2 = np.log2(np.maximum(1e-6, f0_voiced))
        # target ~0.25s smoothing window (keeps melody, reduces vibrato)
        win = int(round(0.25 * float(sr) / 256.0))
        if win < 3:
            win = 3
        if win % 2 == 0:
            win += 1
        win = int(min(win, max(3, (len(f0_log2) // 2) * 2 + 1)))
        f0_smooth = medfilt(f0_log2, kernel_size=win) if len(f0_log2) >= 5 else f0_log2

        residual_cents = 1200.0 * (f0_log2 - f0_smooth)

        # Remove large outliers (likely octave errors / accompaniment)
        inlier_mask = np.abs(residual_cents) <= 300.0
        residual_inliers = residual_cents[inlier_mask]
        f0_voiced_frames = int(len(residual_cents))
        f0_used_frames = int(len(residual_inliers))

        if len(residual_inliers) < 10:
            stability_score = 0
            f0_residual_std_cents = float("nan")
            f0_residual_mad_cents = float("nan")
            stability_note = "too_few_inlier_frames"
        else:
            # Robust dispersion (MAD -> approx std)
            med = float(np.median(residual_inliers))
            mad = float(np.median(np.abs(residual_inliers - med)))
            f0_residual_mad_cents = mad
            f0_residual_std_cents = float(1.4826 * mad)
            stability_note = "ok"

            # Normalize: robust std 12 cents (stable) -> 100.
            # Increase upper bound to avoid 0-clipping for accompaniment-heavy recordings.
            stability_score = 100 - _to_score100(f0_residual_std_cents, lo=12.0, hi=140.0)

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
        "f0_residual_std_cents": f0_residual_std_cents,
        "f0_residual_mad_cents": f0_residual_mad_cents,
        "f0_voiced_frames": float(f0_voiced_frames),
        "f0_used_frames": float(f0_used_frames),
        "stability_note": stability_note,
        "f0_p05_hz": f0_p05,
        "f0_p95_hz": f0_p95,
        "f0_p05_note": hz_to_note(f0_p05),
        "f0_p95_note": hz_to_note(f0_p95),
        "rms_db_median": rms_db_med,
        "rms_db_iqr": rms_db_iqr,
    }

    return MetricResult(score=score, raw=raw)