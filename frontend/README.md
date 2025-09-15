# CV Frontend (Next.js)

A modern Next.js frontend for a computer vision demo that talks to a FastAPI backend.

## Features
- Drag & drop image upload (PNG/JPG, 10MB limit) with preview
- Predict via FastAPI `/predict` with FormData
- Loading spinner, progress bar, retries for network/cold starts
- Results with confidence bars and processing time
- Health checks against `/health`
- Dark theme with blue accents (CSS only)

## Setup

1. Ensure Node.js v22+ is installed. If you use `nvm`:

```bash
nvm install 22
nvm use 22
```

2. Install dependencies and start the dev server:

```bash
cd frontend
npm install
npm run dev
```

3. Backend URL

Create or edit `.env.local`:

```
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Build & Start

```bash
npm run build
npm start
```

## Deploy to Vercel
- Root directory: `frontend`
- Framework preset: Next.js
- Set Environment Variable: `NEXT_PUBLIC_BACKEND_URL=https://your-backend-service.onrender.com`
- Push to `main` to trigger deployments

## Notes on Render Free Tier
- Cold starts can take 15–60s. The UI shows a warning, uses retries and timeouts.
- If you get timeouts on first request, wait a few seconds and try again.

## API
- `POST /predict` body: `multipart/form-data` field `file`
- Expected response shape:

```json
{
  "success": true,
  "predictions": [
    { "class_id": 0, "confidence": 0.95 }
  ],
  "processing_time": 1.23
}
```
