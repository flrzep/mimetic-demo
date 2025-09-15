import React, { useState, useEffect } from 'react';

interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface Prediction {
  class_id: number;
  confidence: number;
  label?: string;
  bbox?: BoundingBox;
}

interface VideoFrame {
  frame_number: number;
  timestamp: number;
  predictions: Prediction[];
}

interface VideoFrameResultsProps {
  frames: VideoFrame[];
  currentTime: number;
  className?: string;
}

export default function VideoFrameResults({ frames, currentTime, className }: VideoFrameResultsProps) {
  const [currentPredictions, setCurrentPredictions] = useState<Prediction[]>([]);
  const [currentFrameNumber, setCurrentFrameNumber] = useState<number | null>(null);

  useEffect(() => {
    // Find predictions for the current video time
    const tolerance = 0.2; // 200ms tolerance for timestamp matching
    
    let bestMatch: VideoFrame | null = null;
    let closestDistance = Infinity;
    
    for (const frame of frames) {
      const distance = Math.abs(frame.timestamp - currentTime);
      if (distance < tolerance && distance < closestDistance) {
        closestDistance = distance;
        bestMatch = frame;
      }
    }
    
    if (bestMatch) {
      setCurrentPredictions(bestMatch.predictions || []);
      setCurrentFrameNumber(bestMatch.frame_number);
    } else {
      setCurrentPredictions([]);
      setCurrentFrameNumber(null);
    }
  }, [frames, currentTime]);

  if (currentPredictions.length === 0) {
    return (
      <div className={`rounded-xl border border-white/10 bg-slate-900/50 p-4 ${className || ''}`}>
        <h3 className="text-sm font-medium text-slate-300 mb-2">Current Frame Predictions</h3>
        <p className="text-xs text-slate-500">No predictions for current frame</p>
      </div>
    );
  }

  return (
    <div className={`rounded-xl border border-white/10 bg-slate-900/50 p-4 ${className || ''}`}>
      <h3 className="text-sm font-medium text-slate-300 mb-2">
        Current Frame Predictions
        {currentFrameNumber !== null && (
          <span className="ml-2 text-xs text-slate-500">
            (Frame {currentFrameNumber} @ {currentTime.toFixed(1)}s)
          </span>
        )}
      </h3>
      
      <div className="space-y-2">
        {currentPredictions.map((pred, index) => (
          <div 
            key={index}
            className="flex items-center justify-between p-2 rounded-lg bg-slate-800/50 border border-white/5"
          >
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-sm text-white font-medium">
                {pred.label || `Class ${pred.class_id}`}
              </span>
            </div>
            
            <div className="flex items-center gap-3 text-xs text-slate-400">
              <span>{(pred.confidence * 100).toFixed(1)}%</span>
              {pred.bbox && (
                <span className="font-mono">
                  {pred.bbox.x},{pred.bbox.y} ({pred.bbox.width}×{pred.bbox.height})
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
      
      <div className="mt-3 text-xs text-slate-500">
        Showing {currentPredictions.length} prediction{currentPredictions.length !== 1 ? 's' : ''} for current frame
      </div>
    </div>
  );
}
