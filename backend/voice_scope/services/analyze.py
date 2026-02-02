from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.analysis import METRIC_KEYS, compute_voice_metrics, metric_labels_jp


MetricKey = Literal["stability", "clarity", "resonance", "brightness", "voiced"]


class MetricsBundle(BaseModel):
    score: dict[MetricKey, int]
    raw: dict[str, Any]


class Insight(BaseModel):
    key: Literal["strong", "weak", "gap", "balance"]
    title: str
    body: str
    highlight: Optional[MetricKey] = None


class Summary(BaseModel):
    overall: int
    match: Optional[int] = None
    reliability_grade: Literal["A", "B", "C"]
    voiced_ratio: float
    pitch_range_note: Optional[str] = None


class AnalyzeResponse(BaseModel):
    source: MetricsBundle
    target: Optional[MetricsBundle] = None
    summary: Summary
    insights: list[Insight]
    balance: float = Field(ge=0.0, le=100.0)
    recommended_highlight_key: Optional[MetricKey] = None


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail="ffmpegが見つかりません。macOSなら `brew install ffmpeg` を実行してください。",
        ) from e
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"ffmpeg変換に失敗しました:\n{e.stderr}") from e


def _convert_to_wav_with_ffmpeg(input_path: Path, output_path: Path, sr: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    _run_ffmpeg(cmd)


def _save_upload_file(upload: UploadFile, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)


def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.asarray(y, dtype=np.float32), int(sr)


def _build_insights(
    source_scores: dict[str, int],
    target_scores: Optional[dict[str, int]],
) -> tuple[list[Insight], float]:
    labels = metric_labels_jp()
    keys: list[MetricKey] = ["stability", "clarity", "resonance", "brightness", "voiced"]

    src_vals = np.array([float(source_scores.get(k, 0)) for k in keys], dtype=float)
    i_strong = int(np.argmax(src_vals))
    i_weak = int(np.argmin(src_vals))

    if target_scores is not None:
        tgt_vals = np.array([float(target_scores.get(k, 0)) for k in keys], dtype=float)
    else:
        tgt_vals = np.array([70.0] * len(keys), dtype=float)

    gaps = tgt_vals - src_vals
    i_gap = int(np.argmax(gaps))

    std = float(np.std(src_vals))
    balance = max(0.0, 100.0 - std * 2.2)

    insights: list[Insight] = [
        Insight(
            key="strong",
            title=f"強み：{labels[keys[i_strong]]}",
            body="自分スコアが最も高い軸。ここを維持しつつ他を底上げ。",
            highlight=keys[i_strong],
        ),
        Insight(
            key="weak",
            title=f"改善候補：{labels[keys[i_weak]]}",
            body="相対的に低い軸。フォーム/環境要因も含めて優先的に見直し。",
            highlight=keys[i_weak],
        ),
        Insight(
            key="gap",
            title=f"理想との差が大きい：{labels[keys[i_gap]]}",
            body="Targetが上回っている差分が最大の軸。比較の優先度が高い。",
            highlight=keys[i_gap],
        ),
        Insight(
            key="balance",
            title=f"バランス：{balance:.0f} / 100",
            body="全体の凸凹の少なさ（計算値）。一部だけ突出していると低下します。",
            highlight=None,
        ),
    ]
    return insights, float(balance)


def _reliability_grade(voiced_ratio: float) -> Literal["A", "B", "C"]:
    if voiced_ratio >= 0.75:
        return "A"
    if voiced_ratio >= 0.55:
        return "B"
    return "C"


async def analyze_files(
    *,
    source: UploadFile,
    target: Optional[UploadFile],
    sr: int,
    consultation_text: Optional[str],
    mode: Optional[str],
) -> AnalyzeResponse:
    if sr <= 0:
        raise HTTPException(status_code=400, detail="sr must be a positive integer")

    # NOTE: consultation_text/mode are accepted for forward-compat; analysis itself is audio-only for now.
    _ = consultation_text
    _ = mode

    keys: list[MetricKey] = ["stability", "clarity", "resonance", "brightness", "voiced"]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        src_suffix = Path(source.filename or "source").suffix or ".bin"
        src_in = tmp / f"source_input{src_suffix}"
        src_wav = tmp / "source_converted.wav"
        _save_upload_file(source, src_in)
        _convert_to_wav_with_ffmpeg(src_in, src_wav, sr=sr)
        y, out_sr = _read_wav_mono(src_wav)
        if out_sr != sr:
            # ffmpeg should force sr; treat mismatch as a signal something went wrong.
            raise HTTPException(status_code=500, detail=f"Unexpected sample rate after conversion: {out_sr}")

        src_metrics = compute_voice_metrics(y, sr)

        tgt_metrics = None
        if target is not None:
            tgt_suffix = Path(target.filename or "target").suffix or ".bin"
            tgt_in = tmp / f"target_input{tgt_suffix}"
            tgt_wav = tmp / "target_converted.wav"
            _save_upload_file(target, tgt_in)
            _convert_to_wav_with_ffmpeg(tgt_in, tgt_wav, sr=sr)
            ty, tsr = _read_wav_mono(tgt_wav)
            if tsr != sr:
                raise HTTPException(status_code=500, detail=f"Unexpected sample rate after conversion: {tsr}")
            tgt_metrics = compute_voice_metrics(ty, sr)

        # Summary (Streamlit互換)
        overall = int(round(float(np.mean([src_metrics.score[k] for k in keys]))))
        match: Optional[int] = None
        if tgt_metrics is not None:
            diffs = [abs(src_metrics.score[k] - tgt_metrics.score[k]) for k in keys]
            match = int(round(100.0 - float(np.mean(diffs))))
            match = int(np.clip(match, 0, 100))

        voiced_ratio = float(src_metrics.raw.get("voiced_ratio", 0.0) or 0.0)
        reliability_grade = _reliability_grade(voiced_ratio)
        pitch_range_note: Optional[str] = None
        p05 = src_metrics.raw.get("f0_p05_note")
        p95 = src_metrics.raw.get("f0_p95_note")
        if isinstance(p05, str) and isinstance(p95, str) and p05 != "-" and p95 != "-":
            pitch_range_note = f"{p05}–{p95}"

        insights, balance = _build_insights(src_metrics.score, tgt_metrics.score if tgt_metrics else None)
        recommended_highlight_key: Optional[MetricKey]
        if tgt_metrics is not None:
            recommended_highlight_key = insights[2].highlight  # gap
        else:
            recommended_highlight_key = insights[1].highlight  # weak

        return AnalyzeResponse(
            source=MetricsBundle(score=src_metrics.score, raw=src_metrics.raw),
            target=MetricsBundle(score=tgt_metrics.score, raw=tgt_metrics.raw) if tgt_metrics else None,
            summary=Summary(
                overall=overall,
                match=match,
                reliability_grade=reliability_grade,
                voiced_ratio=voiced_ratio,
                pitch_range_note=pitch_range_note,
            ),
            insights=insights,
            balance=balance,
            recommended_highlight_key=recommended_highlight_key,
        )
