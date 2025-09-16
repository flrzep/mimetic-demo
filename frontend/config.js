const isProduction = process.env.NODE_ENV === 'production' || 
                    typeof window !== 'undefined' && window.location.hostname !== 'localhost';

export const API_BASE_URL = isProduction 
  ? (process.env.NEXT_PUBLIC_BACKEND_URL || 'https://memetic-demo-backend.onrender.com')
  : 'http://localhost:8000';

export const REQUEST_TIMEOUT = isProduction ? 60000 : 30000; // Production timeout is longer
export const WS_RECONNECT_INTERVAL = 5000;
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

export const ENDPOINTS = {
  predict: '/predict',
  predictVideo: '/predict_video',
  health: '/health',
  ws: '/ws',
  wsVideo: '/ws/video'
};
