import React from 'react';
import { Settings, Info } from 'lucide-react';

interface VideoMetadata {
  name: string;
  size: number;
  type: string;
  duration?: number;
  width?: number;
  height?: number;
  fps?: number;
}

interface VideoOptionsProps {
  frameInterval: number;
  maxFrames: number | null;
  onFrameIntervalChange: (value: number) => void;
  onMaxFramesChange: (value: number | null) => void;
  videoMetadata?: VideoMetadata;
  isProcessing?: boolean;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

const VideoOptions: React.FC<VideoOptionsProps> = ({
  frameInterval,
  maxFrames,
  onFrameIntervalChange,
  onMaxFramesChange,
  videoMetadata,
  isProcessing = false
}) => {
  const handleFrameIntervalChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value, 10);
    if (value >= 1 && value <= 60) {
      onFrameIntervalChange(value);
    }
  };

  const handleMaxFramesChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value;
    if (value === '') {
      onMaxFramesChange(null);
    } else {
      const numValue = parseInt(value, 10);
      if (numValue >= 1) {
        onMaxFramesChange(numValue);
      }
    }
  };

  const estimatedFrames = videoMetadata?.duration && videoMetadata?.fps 
    ? Math.ceil((videoMetadata.duration * videoMetadata.fps) / frameInterval)
    : null;

  const finalFrameCount = maxFrames && estimatedFrames 
    ? Math.min(maxFrames, estimatedFrames)
    : estimatedFrames || maxFrames;

  return (
    <div className="bg-slate-800/50 border border-white/10 rounded-lg p-4 space-y-4">
      <div className="flex items-center gap-2 text-white">
        <Settings className="w-4 h-4" />
        <h3 className="font-medium">Processing Options</h3>
      </div>

      {/* Video Metadata */}
      {videoMetadata && (
        <div className="bg-slate-900/50 rounded-lg p-3 space-y-2">
          <div className="flex items-center gap-2 text-slate-300 text-sm">
            <Info className="w-4 h-4" />
            <span className="font-medium">Video Information</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs text-slate-400">
            <div>
              <span className="text-slate-300">Name:</span>
              <div className="truncate" title={videoMetadata.name}>
                {videoMetadata.name}
              </div>
            </div>
            <div>
              <span className="text-slate-300">Size:</span>
              <div>{formatBytes(videoMetadata.size)}</div>
            </div>
            {videoMetadata.duration && (
              <div>
                <span className="text-slate-300">Duration:</span>
                <div>{formatDuration(videoMetadata.duration)}</div>
              </div>
            )}
            {videoMetadata.width && videoMetadata.height && (
              <div>
                <span className="text-slate-300">Resolution:</span>
                <div>{videoMetadata.width}×{videoMetadata.height}</div>
              </div>
            )}
            {videoMetadata.fps && (
              <div>
                <span className="text-slate-300">FPS:</span>
                <div>{videoMetadata.fps.toFixed(1)}</div>
              </div>
            )}
            <div>
              <span className="text-slate-300">Format:</span>
              <div>{videoMetadata.type || 'Unknown'}</div>
            </div>
          </div>
        </div>
      )}

      {/* Processing Options */}
      <div className="space-y-3">
        <div>
          <label className="block text-sm text-slate-300 mb-2">
            Frame Interval
            <span className="text-xs text-slate-400 ml-2">(process every Nth frame)</span>
          </label>
          <input
            type="range"
            min="1"
            max="60"
            step="1"
            value={frameInterval}
            onChange={handleFrameIntervalChange}
            disabled={isProcessing}
            className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed slider"
          />
          <div className="flex justify-between text-xs text-slate-400 mt-1">
            <span>1 (every frame)</span>
            <span className="text-white font-medium">{frameInterval}</span>
            <span>60 (every 60th frame)</span>
          </div>
        </div>

        <div>
          <label className="block text-sm text-slate-300 mb-2">
            Max Frames
            <span className="text-xs text-slate-400 ml-2">(limit total frames processed)</span>
          </label>
          <div className="flex items-center gap-2">
            <input
              type="number"
              min="1"
              max="1000"
              value={maxFrames || ''}
              onChange={handleMaxFramesChange}
              placeholder="No limit"
              disabled={isProcessing}
              className="flex-1 bg-slate-700 border border-slate-600 rounded-md px-3 py-2 text-white text-sm placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <button
              type="button"
              onClick={() => onMaxFramesChange(null)}
              disabled={isProcessing || maxFrames === null}
              className="px-3 py-2 text-xs bg-slate-600 hover:bg-slate-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-md text-white"
            >
              Clear
            </button>
          </div>
        </div>

        {/* Frame Count Estimation */}
        {finalFrameCount && (
          <div className="bg-blue-900/20 border border-blue-500/20 rounded-lg p-3">
            <div className="text-sm text-blue-300">
              <strong>Estimated frames to process:</strong> {finalFrameCount.toLocaleString()}
            </div>
            {estimatedFrames && maxFrames && finalFrameCount < estimatedFrames && (
              <div className="text-xs text-blue-400 mt-1">
                Limited by max frames setting ({maxFrames.toLocaleString()} frames)
              </div>
            )}
          </div>
        )}
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e293b;
        }

        .slider::-moz-range-thumb {
          height: 16px;
          width: 16px;
          border-radius: 50%;
          background: #3b82f6;
          cursor: pointer;
          border: 2px solid #1e293b;
        }
      `}</style>
    </div>
  );
};

export default VideoOptions;