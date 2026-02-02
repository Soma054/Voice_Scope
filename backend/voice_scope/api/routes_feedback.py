from __future__ import annotations

from fastapi import APIRouter

from backend.voice_scope.services.feedback import FeedbackRequest, build_feedback

router = APIRouter()


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    return build_feedback(req)
