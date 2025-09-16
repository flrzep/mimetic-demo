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
    <div className="grid gap-3">
      <div className="rounded-2xl border-2 border-dashed border-white/10 p-4 sm:p-6 transition min-h-[180px] sm:min-h-[220px] flex items-center justify-center">
        {hasCamera ? (
          <video 
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="max-w-full max-h-72 sm:max-h-96 rounded-xl border border-white/10 bg-slate-950 object-contain"
            style={{ aspectRatio: 'auto' }}
            aria-label="Live camera stream"
          />
        ) : (
          <div className="grid place-items-center gap-2 text-center">
            <div className="w-12 h-12 sm:w-14 sm:h-14 grid place-items-center text-green-500 bg-green-500/20 rounded-full">
              <Camera className="w-7 h-7 sm:w-9 sm:h-9" />
            </div>
            <p className="font-semibold text-sm sm:text-base">Camera Stream</p>
            <p className="text-slate-400 -mt-1 text-xs sm:text-sm max-w-md">
              Click &quot;Start Camera&quot; to begin real-time video processing
            </p>
          </div>
        )}
      </div>

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
              onClick={isStreaming ? onStop : onStart}
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
              className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 text-sm sm:text-base" 
              onClick={stopCamera}
              aria-label="Stop camera"
            >
              <Square className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span>Stop Camera</span>
            </button>
          </>
        )}
        
        <button 
          type="button" 
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 text-sm sm:text-base" 
          onClick={onSettings}
          aria-label="Stream settings"
        >
          <Settings className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span className="sm:inline hidden">Settings</span>
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
