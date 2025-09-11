export const API_BASE_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
export const REQUEST_TIMEOUT = process.env.NODE_ENV === 'production' ? 60000 : 30000;
export const WS_RECONNECT_INTERVAL = 5000;
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export const ENDPOINTS = {
  predict: '/predict',
  predictBase64: '/predict-base64',
  health: '/health',
  ws: '/ws'
};
