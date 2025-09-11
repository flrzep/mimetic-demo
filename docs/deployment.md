# Deployment Guide

## Modal Service
```
cd modal-service
modal setup
modal deploy modal_service.py
```

## Backend (Render)
- Create a new Web Service
- Root directory: `backend`
- Build: `pip install -r requirements.txt`
- Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- Set env vars (MODAL_ENDPOINT_URL, MODAL_API_KEY)

## Frontend (Vercel)
- Root directory: `frontend`
- Framework preset: Create React App
- Env var: `REACT_APP_BACKEND_URL`
