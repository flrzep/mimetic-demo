import axios from 'axios';
import { API_BASE_URL, ENDPOINTS, REQUEST_TIMEOUT, WS_RECONNECT_INTERVAL } from '../config';

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT,
});

api.interceptors.response.use(
  (res) => res,
  (error) => {
    // Normalize network errors
    if (error.message === 'Network Error' && !error.response) {
      error.message = 'Network error: Unable to reach backend.';
    }
    return Promise.reject(error);
  }
);

export async function checkHealth() {
  try {
    const res = await api.get(ENDPOINTS.health, { timeout: 5000 });
    const ok = res.status === 200 && (res.data?.status === 'ok' || res.data?.success);
    return { ok, message: ok ? '' : 'Backend health check failed' };
  } catch (e) {
    return { ok: false, message: 'Backend unreachable' };
  }
}

// Image prediction with progress tracking
export const predictImage = async (imageFile, onProgress) => {
  const formData = new FormData();
  formData.append('file', imageFile);
  return api.post(ENDPOINTS.predict, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: onProgress,
  });
};

// WebSocket connection for real-time updates with basic reconnect
export const connectWebSocket = (onMessage, onError) => {
  const wsUrl = API_BASE_URL.replace('http', 'ws') + ENDPOINTS.ws;
  let ws = new WebSocket(wsUrl);
  let shouldReconnect = true;
  let timeoutId = null;

  const cleanup = () => {
    if (timeoutId) clearTimeout(timeoutId);
  };

  ws.onmessage = onMessage;
  ws.onerror = (e) => {
    onError && onError(e);
    ws.close();
  };
  ws.onclose = () => {
    if (!shouldReconnect) return;
    timeoutId = setTimeout(() => {
      ws = connectWebSocket(onMessage, onError);
    }, WS_RECONNECT_INTERVAL);
  };

  ws.destroy = () => {
    shouldReconnect = false;
    cleanup();
    try { ws.close(); } catch {}
  };

  return ws;
};

// Health check with retry logic
export const healthCheck = async (retries = 3) => {
  for (let i = 0; i < retries; i++) {
    try {
      const response = await api.get(ENDPOINTS.health);
      return response.data;
    } catch (error) {
      if (i === retries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
    }
  }
};
