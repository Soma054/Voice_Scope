from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.voice_scope.api.routes_analyze import router as analyze_router
from backend.voice_scope.api.routes_feedback import router as feedback_router


def create_app() -> FastAPI:
	app = FastAPI(title="Voice-Scope Backend", version="v1")

	cors_origins = os.environ.get("VOICE_SCOPE_CORS_ORIGINS", "http://localhost:5173").split(",")
	cors_origins = [o.strip() for o in cors_origins if o.strip()]

	app.add_middleware(
		CORSMiddleware,
		allow_origins=cors_origins or ["*"],
		allow_credentials=False,
		allow_methods=["*"],
		allow_headers=["*"],
	)

	@app.get("/health")
	def health() -> dict:
		return {"status": "ok"}

	app.include_router(analyze_router, prefix="/v1", tags=["analyze"])
	app.include_router(feedback_router, prefix="/v1", tags=["feedback"])

	return app


app = create_app()

