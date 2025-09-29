# Modal Service - Auto-Deploy Setup

This Modal service provides AI inference for the memetic-demo app with automatic deployment from GitHub.

## 🔗 Repository Auto-Deployment (Recommended)

Modal supports automatic deployment directly from your GitHub repository, similar to Vercel and Render.

### **Setup Steps:**

1. **Connect Repository in Modal Dashboard**
   - Go to [Modal Dashboard](https://modal.com/apps)
   - Click "New App" → "Connect GitHub Repository"
   - Select repository: `flrzep/memetic-demo`
   - Set deployment path: `modal-service/modal_service.py`

2. **Configure Auto-Deploy**
   - Branch: `main`
   - Auto-deploy: ✅ Enabled
   - Modal will provide stable URLs automatically

3. **Update Backend Environment**
   ```env
   USE_MOCK_MODAL=false
   MODAL_BASE_URL=https://your-modal-url.modal.run
   MODAL_WEBRTC_URL=https://your-webrtc-url.modal.run
   ```

### **Deployment Process:**
- Push to `main` → Modal automatically deploys
- Check Modal Dashboard for deployment status
- Use provided URLs in your backend

## 🧪 Local Development

```bash
pip install -r requirements.txt
uvicorn modal_service:app --reload --port 9000
```

## 📋 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Image prediction
- `WebSocket /ws/{client_id}` - WebRTC signaling

No GitHub Actions needed - Modal handles everything automatically!

---

## 📦 Managing Models & Config in Modal Volume

Use the helper script `manage_models.py` and the quick guide in `MANAGE_MODELS.md` to upload and validate model files and `model_config.json`.

Quick commands (Windows PowerShell):

```powershell
# Authenticate (one-time)
py -m modal token set

# From repo root
cd modal-service

# Upload only the config after local edits
py -m modal run manage_models.py::upload_model_config

# Upload all local model files under modal-service/models
py -m modal run manage_models.py::upload_local_models

# Full setup: upload models + config, then validate
py -m modal run manage_models.py::setup_modal_storage

# Verify and validate
py -m modal run manage_models.py::list_models
py -m modal run manage_models.py::validate_model_config
```

See `MANAGE_MODELS.md` for more details.
