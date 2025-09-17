import axios from 'axios';
import { API_BASE_URL, ENDPOINTS, REQUEST_TIMEOUT, WS_RECONNECT_INTERVAL } from '../config';
import { supabase } from '../lib/supabase';

const REQUIRE_AUTH = process.env.NEXT_PUBLIC_REQUIRE_AUTH === 'true';

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

// Inject Supabase auth token if available and required
api.interceptors.request.use(async (config) => {
  if (REQUIRE_AUTH && supabase) {
    try {
      const { data } = await supabase.auth.getSession();
      const accessToken = data?.session?.access_token;
      if (accessToken) {
        config.headers = config.headers || {};
        config.headers['Authorization'] = `Bearer ${accessToken}`;
      }
    } catch {}
  }
  return config;
});

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

// Video prediction with progress tracking
export const predictVideo = async (videoFile, options = {}, onProgress) => {
  // Convert video file to base64
  const videoBase64 = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Remove the data:video/...;base64, prefix to get just the base64 data
      const base64Data = reader.result.split(',')[1];
      resolve(base64Data);
    };
    reader.onerror = reject;
    reader.readAsDataURL(videoFile);
  });
  
  // Prepare request payload matching backend expectations
  const payload = {
    video_data: videoBase64,
    filename: videoFile.name,
    video_codec: options.video_codec || "h264",
    audio_codec: options.audio_codec || "none", 
    return_url: options.return_url || false
  };
  
  return api.post(ENDPOINTS.predictVideo, payload, {
    headers: { 'Content-Type': 'application/json' },
    onUploadProgress: onProgress,
    timeout: 300000, // 5 minutes for video processing
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
