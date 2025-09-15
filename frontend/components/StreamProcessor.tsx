import React, { useRef, useState, useCallback } from 'react';
import { Camera, Square, Play, Settings } from 'lucide-react';

type Props = {
  onStart: () => void;
  onStop: () => void;
  isStreaming: boolean;
  onSettings: () => void;
};

const StreamProcessor: React.FC<Props> = ({ onStart, onStop, isStreaming, onSettings }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [hasCamera, setHasCamera] = useState(false);

  const initializeCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 }, 
        audio: false 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setHasCamera(true);
      }
    } catch (err) {
      console.error('Camera access denied:', err);
      setHasCamera(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setHasCamera(false);
    }
  }, []);

  return (
    <div className="grid gap-4">
      <div className="rounded-2xl border border-white/10 bg-slate-950 p-4 min-h-[300px] flex items-center justify-center">
        {hasCamera ? (
          <video 
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="max-w-full max-h-[400px] rounded-lg border border-white/10"
            aria-label="Live camera stream"
          />
        ) : (
          <div className="grid place-items-center gap-4 text-center">
            <div className="w-16 h-16 grid place-items-center text-green-500 bg-green-500/20 rounded-full">
              <Camera size={32} />
            </div>
            <div>
              <h3 className="font-semibold mb-2">Camera Stream</h3>
              <p className="text-slate-400 text-sm max-w-md">
                Click "Start Camera" to begin real-time video processing
              </p>
            </div>
          </div>
        )}
      </div>

      <div className="flex items-center justify-between">
        <div className="flex gap-2">
          {!hasCamera ? (
            <button 
              type="button" 
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-green-500 hover:bg-green-600 text-white" 
              onClick={initializeCamera}
              aria-label="Start camera"
            >
              <Camera size={16} />
              <span>Start Camera</span>
            </button>
          ) : (
            <>
              <button 
                type="button" 
                className={`inline-flex items-center gap-2 px-4 py-2 rounded-md text-white ${
                  isStreaming 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-green-500 hover:bg-green-600'
                }`}
                onClick={isStreaming ? onStop : onStart}
                aria-label={isStreaming ? 'Stop processing' : 'Start processing'}
              >
                {isStreaming ? (
                  <>
                    <Square size={16} />
                    <span>Stop Processing</span>
                  </>
                ) : (
                  <>
                    <Play size={16} />
                    <span>Start Processing</span>
                  </>
                )}
              </button>
              <button 
                type="button" 
                className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10" 
                onClick={stopCamera}
                aria-label="Stop camera"
              >
                <Square size={16} />
                <span>Stop Camera</span>
              </button>
            </>
          )}
        </div>

        <button 
          type="button" 
          className="inline-flex items-center gap-2 px-3 py-2 rounded-md border border-white/10 hover:bg-white/10" 
          onClick={onSettings}
          aria-label="Stream settings"
        >
          <Settings size={16} />
        </button>
      </div>

      {isStreaming && (
        <div className="rounded-lg border border-green-200/30 bg-green-500/10 p-3">
          <div className="flex items-center gap-2 text-green-100">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            <span className="text-sm">Live processing active</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default StreamProcessor;
