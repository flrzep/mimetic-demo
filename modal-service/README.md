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
