from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile

from backend.voice_scope.services.analyze import analyze_files

router = APIRouter()


@router.post("/analyze")
async def analyze(
    source: UploadFile = File(...),
    target: Optional[UploadFile] = File(None),
    sr: int = Form(44100),
    consultation_text: Optional[str] = Form(None),
    mode: Optional[str] = Form(None),
):
    return await analyze_files(
        source=source,
        target=target,
        sr=sr,
        consultation_text=consultation_text,
        mode=mode,
    )
