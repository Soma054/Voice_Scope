from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_AUDIO_EXTS = [
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus",
    "webm", "mp4", "mov", "aiff", "aif", "caf", "wma",
]

class AudioLoadError(RuntimeError):
    pass

def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise AudioLoadError(
            "ffmpegが見つかりません。macOSなら `brew install ffmpeg` を実行してください。"
        ) from e
    except subprocess.CalledProcessError as e:
        # ffmpegのstderrが一番役に立つ
        raise AudioLoadError(f"ffmpeg変換に失敗しました:\n{e.stderr}") from e

def convert_to_wav_with_ffmpeg(input_path: Path, output_path: Path, sr: int = 44100) -> None:
    """
    Convert any audio/video file to mono wav using ffmpeg.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        str(output_path),
    ]
    _run_ffmpeg(cmd)

def load_uploaded_audio(uploaded_file, sr: int = 44100) -> tuple[np.ndarray, int]:
    """
    StreamlitのUploadedFileを受け取り、ffmpegでwavに変換してから読み込む。
    戻り値: (y: float32 mono, sr)
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    if not suffix:
        suffix = ".bin"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in_path = tmpdir / f"input{suffix}"
        out_path = tmpdir / "output.wav"

        in_path.write_bytes(uploaded_file.getvalue())
        convert_to_wav_with_ffmpeg(in_path, out_path, sr=sr)

        y, out_sr = sf.read(out_path, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y, out_sr