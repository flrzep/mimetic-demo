import React, { useRef, useState, useCallback, useEffect } from 'react';
import { Camera, Square, Play, Settings, Wifi, WifiOff } from 'lucide-react';
import ModalWebRtcClient from '../utils/webrtc';

interface Prediction {
  class_id: number;
  confidence: number;
  label?: string;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

type Props = {
  onStart: () => void;
  onStop: () => void;
  isStreaming: boolean;
  onSettings: () => void;
};

const StreamProcessor: React.FC<Props> = ({ onStart, onStop, isStreaming, onSettings }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const remoteVideoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const webrtcClientRef = useRef<ModalWebRtcClient | null>(null);
  
  const [hasCamera, setHasCamera] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<string>('disconnected');
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [error, setError] = useState<string>('');

  // Initialize WebRTC client
  useEffect(() => {
    webrtcClientRef.current = new ModalWebRtcClient();
    const client = webrtcClientRef.current;

    // Set up event listeners
    client.addEventListener('status', (event: any) => {
      setConnectionStatus(event.detail.message);
    });

    client.addEventListener('localStream', (event: any) => {
      if (videoRef.current) {
        videoRef.current.srcObject = event.detail.stream;
      }
    });

    client.addEventListener('remoteStream', (event: any) => {
      if (remoteVideoRef.current) {
        remoteVideoRef.current.srcObject = event.detail.stream;
      }
    });

    client.addEventListener('prediction', (event: any) => {
      setPredictions(event.detail.predictions || []);
    });

    client.addEventListener('error', (event: any) => {
      setError(event.detail.error?.message || 'Unknown error');
    });

    client.addEventListener('connected', () => {
      setConnectionStatus('connected');
    });

    client.addEventListener('streamingStarted', () => {
      onStart();
    });

    client.addEventListener('streamingStopped', () => {
      onStop();
      setHasCamera(false);
    });

    return () => {
      client.stopStreaming();
    };
  }, [onStart, onStop]);

  // Draw predictions overlay
  const drawPredictions = useCallback(() => {
    const canvas = canvasRef.current;
    const video = remoteVideoRef.current;
    
    if (!canvas || !video || predictions.length === 0) {
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth || video.clientWidth;
    canvas.height = video.videoHeight || video.clientHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes
    predictions.forEach((pred) => {
      if (!pred.bbox) return;

      const { x, y, width, height } = pred.bbox;
      
      // Draw bounding box
      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, width, height);

      // Draw label
      const label = `${pred.label || `Class ${pred.class_id}`}: ${pred.confidence.toFixed(2)}`;
      
      const fontSize = Math.max(12, Math.floor(canvas.height * 0.02));
      ctx.font = `${fontSize}px Arial`;
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;

      const labelY = y > textHeight + 10 ? y - 5 : y + height + textHeight + 5;

      // Draw background for text
      ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
      ctx.fillRect(x, labelY - textHeight, textWidth + 8, textHeight + 4);

      // Draw text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x + 4, labelY - 4);
    });
  }, [predictions]);

  // Update overlay when predictions change
  useEffect(() => {
    drawPredictions();
  }, [drawPredictions]);

  const initializeCamera = useCallback(async () => {
    const client = webrtcClientRef.current;
    if (!client) return;

    try {
      setError('');
      await client.startWebcam();
      setHasCamera(true);
    } catch (err) {
      console.error('Camera access denied:', err);
      setError(err instanceof Error ? err.message : 'Camera access denied');
      setHasCamera(false);
    }
  }, []);

  const startStreaming = useCallback(async () => {
    const client = webrtcClientRef.current;
    if (!client || !hasCamera) return;

    try {
      setError('');
      await client.startStreaming();
    } catch (err) {
      console.error('Failed to start streaming:', err);
      setError(err instanceof Error ? err.message : 'Failed to start streaming');
    }
  }, [hasCamera]);

  const stopStreaming = useCallback(async () => {
    const client = webrtcClientRef.current;
    if (!client) return;

    try {
      await client.stopStreaming();
      setPredictions([]);
    } catch (err) {
      console.error('Failed to stop streaming:', err);
    }
  }, []);

  const getConnectionIcon = () => {
    if (connectionStatus === 'connected') {
      return <Wifi className="w-4 h-4 text-green-500" />;
    }
    return <WifiOff className="w-4 h-4 text-red-500" />;
  };

  return (
    <div className="grid gap-3">
      {/* Status bar */}
      <div className="flex items-center justify-between p-2 rounded-lg bg-slate-800/50">
        <div className="flex items-center gap-2">
          {getConnectionIcon()}
          <span className="text-sm text-slate-300">
            {connectionStatus === 'connected' ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        {predictions.length > 0 && (
          <span className="text-sm text-green-400">
            {predictions.length} detection{predictions.length !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Video streams */}
      <div className="grid gap-4">
        {/* Local camera stream */}
        <div className="rounded-2xl border-2 border-dashed border-white/10 p-4 sm:p-6 transition min-h-[180px] sm:min-h-[220px] flex items-center justify-center">
          {hasCamera ? (
            <div className="relative max-w-full">
              <video 
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="max-w-full max-h-72 sm:max-h-96 rounded-xl border border-white/10 bg-slate-950 object-contain"
                style={{ aspectRatio: 'auto' }}
                aria-label="Local camera stream"
              />
              <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded-md text-xs text-white">
                Local Camera
              </div>
            </div>
          ) : (
            <div className="grid place-items-center gap-2 text-center">
              <div className="w-12 h-12 sm:w-14 sm:h-14 grid place-items-center text-green-500 bg-green-500/20 rounded-full">
                <Camera className="w-7 h-7 sm:w-9 sm:h-9" />
              </div>
              <p className="font-semibold text-sm sm:text-base">WebRTC Stream</p>
              <p className="text-slate-400 -mt-1 text-xs sm:text-sm max-w-md">
                Click &quot;Start Camera&quot; to begin real-time video processing with Modal
              </p>
            </div>
          )}
        </div>

        {/* Remote processed stream */}
        {isStreaming && (
          <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-4 shadow-xl">
            <h3 className="text-lg font-semibold text-white mb-4">Live Processed Stream</h3>
            <div className="relative flex justify-center">
              <div className="relative max-w-full">
                <video 
                  ref={remoteVideoRef}
                  autoPlay
                  playsInline
                  muted
                  className="max-w-full max-h-72 sm:max-h-96 rounded-xl border border-white/10 bg-slate-950 object-contain"
                  style={{ aspectRatio: 'auto' }}
                  aria-label="Processed stream from Modal"
                />
                
                {/* Prediction overlay canvas */}
                <canvas
                  ref={canvasRef}
                  className="absolute top-0 left-0 pointer-events-none rounded-xl"
                  style={{ 
                    width: '100%', 
                    height: '100%',
                    objectFit: 'contain'
                  }}
                />
                
                <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded-md text-xs text-white">
                  Modal AI Processing
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Error display */}
      {error && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
          <p className="text-red-400 text-sm">{error}</p>
        </div>
      )}

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-2">
        {!hasCamera ? (
          <button 
            type="button" 
            className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-green-500 hover:bg-green-600 text-white text-sm sm:text-base" 
            onClick={initializeCamera}
            aria-label="Start camera"
          >
            <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <span>Start Camera</span>
          </button>
        ) : (
          <>
            <button 
              type="button" 
              className={`inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md text-white text-sm sm:text-base ${
                isStreaming 
                  ? 'bg-red-500 hover:bg-red-600' 
                  : 'bg-green-500 hover:bg-green-600'
              }`}
              onClick={isStreaming ? stopStreaming : startStreaming}
              aria-label={isStreaming ? 'Stop processing' : 'Start processing'}
            >
              {isStreaming ? (
                <>
                  <Square className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>Stop Processing</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>Start Processing</span>
                </>
              )}
            </button>
            
            <button 
              type="button" 
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-slate-600 hover:bg-slate-700 text-white text-sm sm:text-base"
              onClick={onSettings}
              aria-label="Settings"
            >
              <Settings className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span>Settings</span>
            </button>
          </>
        )}
      </div>
    </div>
  );
};

export default StreamProcessor;
