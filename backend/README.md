# Voice-Scope Backend (FastAPI)

## Prerequisites
- ffmpeg installed (macOS): `brew install ffmpeg`

## Install
This repo already has a `venv/` for the Streamlit prototype. You can reuse it or create a dedicated venv.

```bash
pip install -r backend/requirements.txt
```

## Run
```bash
uvicorn backend.voice_scope.api:app --reload --port 8000
```

## Quick test (curl)

### 1) Health
```bash
curl -sS http://127.0.0.1:8000/health
```

### 2) Analyze (multipart)
```bash
curl -sS -X POST http://127.0.0.1:8000/v1/analyze \
	-F "source=@/path/to/source.wav" \
	-F "target=@/path/to/target.wav" \
	-F "sr=44100"
```

### 3) Feedback (json)
`/v1/analyze` のレスポンスを `analysis` として渡します。
```bash
curl -sS -X POST http://127.0.0.1:8000/v1/feedback \
	-H 'Content-Type: application/json' \
	-d '{"consultation_text":"悩み(任意)","analysis":{}}'
```

## Endpoints
- `GET /health`
- `POST /v1/analyze` (multipart: `source` required, `target` optional, `sr` optional)
- `POST /v1/feedback` (json)
