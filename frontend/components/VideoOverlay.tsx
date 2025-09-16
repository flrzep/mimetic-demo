import React, { useRef, useEffect, useState, useCallback } from 'react';

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

interface VideoOverlayProps {
  videoSrc: string;
  frames: VideoFrame[];
  className?: string;
  onError?: (error: string) => void;
  onTimeUpdate?: (currentTime: number) => void;
}

export default function VideoOverlay({ videoSrc, frames, className, onError, onTimeUpdate }: VideoOverlayProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<number>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 0 });
  
  // Create lookup for frame predictions by timestamp
  const predictionLookup = useCallback(() => {
    const lookup = new Map<number, Prediction[]>();
    frames.forEach(frame => {
      if (frame.predictions && frame.predictions.length > 0) {
        // Round timestamp to nearest 0.1 seconds for more reliable matching
        const roundedTime = Math.round(frame.timestamp * 10) / 10;
        lookup.set(roundedTime, frame.predictions);
      }
    });
    return lookup;
  }, [frames]);

  const predictions = predictionLookup();

  // Draw bounding boxes on canvas
  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) {
      console.log('Canvas or video not available for overlay');
      return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.log('Canvas context not available');
      return;
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Get current video time and find matching predictions
    const currentVideoTime = video.currentTime;
    const roundedTime = Math.round(currentVideoTime * 10) / 10;
    
    // Look for predictions within a small time window (±0.2 seconds for better matching)
    let currentPredictions: Prediction[] = [];
    for (let offset = -0.2; offset <= 0.2; offset += 0.1) {
      const searchTime = Math.round((roundedTime + offset) * 10) / 10;
      if (predictions.has(searchTime)) {
        currentPredictions = predictions.get(searchTime) || [];
        break;
      }
    }

    if (currentPredictions.length === 0) return;

    // Calculate scale factors
    const scaleX = canvas.width / video.videoWidth;
    const scaleY = canvas.height / video.videoHeight;

    // Draw bounding boxes
    currentPredictions.forEach((pred, index) => {
      if (!pred.bbox) {
        return;
      }

      const { x, y, width, height } = pred.bbox;
      
      // Scale coordinates to canvas size
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;

      // Draw bounding box with bright color for visibility
      ctx.strokeStyle = '#00ff00'; // Bright green
      ctx.lineWidth = 3; // Thicker line
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

      // Draw label with background
      const label = `${pred.label || `Class ${pred.class_id}`}: ${pred.confidence.toFixed(2)}`;
      
      // Measure text
      const fontSize = Math.floor(video.videoHeight * 0.02); // 2% of video height
      ctx.font = `${fontSize}px Arial`; // Bigger font
      const textMetrics = ctx.measureText(label);
      const textWidth = textMetrics.width;
      const textHeight = fontSize;

      // Position label above bounding box, or below if too close to top
      const labelY = scaledY > textHeight + 10 ? scaledY - 5 : scaledY + scaledHeight + textHeight + 5;

      // Draw background rectangle for text
      ctx.fillStyle = 'rgba(0, 255, 0, 0.9)'; // More opaque
      ctx.fillRect(scaledX, labelY - textHeight, textWidth + 8, textHeight + 4);

      // Draw text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, scaledX + 4, labelY - 4);
    });
  }, [predictions]);

  // Animation frame callback (following MDN pattern for smoother updates)
  
  const timerCallback = useCallback(() => {
    const video = videoRef.current;
    if (!video || video.paused || video.ended) {
      return;
    }
    
    // Draw overlay for current frame
    drawOverlay();
    
    // Schedule next frame (following MDN pattern)
    animationFrameRef.current = requestAnimationFrame(timerCallback);
  }, [drawOverlay]);

  // Handle video events
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const handleLoadedMetadata = () => {
      const { videoWidth, videoHeight } = video;
      setVideoDimensions({ width: videoWidth, height: videoHeight });
      
      // Set canvas size to match video
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = videoWidth;
        canvas.height = videoHeight;
      }
    };

    const handleTimeUpdate = () => {
      const currentVideoTime = video.currentTime;
      setCurrentTime(currentVideoTime);
      onTimeUpdate?.(currentVideoTime);
      // Note: drawOverlay is now handled by animation frame for smoother updates
    };

    const handlePlay = () => {
      setIsPlaying(true);
      // Start animation frame loop when video plays (MDN pattern)
      timerCallback();
    };
    
    const handlePause = () => {
      setIsPlaying(false);
      // Cancel animation frame when paused
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
    
    const handleError = (e: Event) => {
      const error = video.error;
      const errorMsg = error ? `Video error: ${error.message || 'Unknown error'}` : 'Video loading failed';
      onError?.(errorMsg);
    };

    video.addEventListener('loadedmetadata', handleLoadedMetadata);
    video.addEventListener('timeupdate', handleTimeUpdate);
    video.addEventListener('play', handlePlay);
    video.addEventListener('pause', handlePause);
    video.addEventListener('error', handleError);

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata);
      video.removeEventListener('timeupdate', handleTimeUpdate);
      video.removeEventListener('play', handlePlay);
      video.removeEventListener('pause', handlePause);
      video.removeEventListener('error', handleError);
      
      // Cleanup animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [drawOverlay, onError, timerCallback]);

  // Redraw overlay when predictions change
  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  return (
    <div className={`relative inline-block ${className || ''}`} style={{ maxWidth: '100%' }}>
      {/* Video element */}
      <video
        ref={videoRef}
        src={videoSrc}
        controls
        playsInline
        {...({ 'webkit-playsinline': 'true' } as any)}
        {...({ 'x-webkit-airplay': 'allow' } as any)}
        controlsList="nodownload nofullscreen noremoteplayback"
        disablePictureInPicture
        className="block w-auto h-auto max-w-full object-contain"
        style={{ 
          display: 'block',
          maxHeight: '80vh', // Use viewport height for better responsiveness
          aspectRatio: 'auto'
        }}
        preload="metadata"
      />
      
      {/* Overlay canvas */}
      <canvas
        ref={canvasRef}
        className="absolute top-0 left-0 pointer-events-none"
        style={{ 
          width: '100%', 
          height: '100%',
          objectFit: 'contain'
        }}
      />      {/* Debug info */}
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white text-xs p-2 rounded max-w-xs">
          <div>Time: {currentTime.toFixed(1)}s</div>
          <div>Dimensions: {videoDimensions.width}x{videoDimensions.height}</div>
          <div>Total Frames: {frames.length}</div>
          <div>Prediction Map Size: {predictions.size}</div>
          <div>Sample Times: {Array.from(predictions.keys()).slice(0, 3).join(', ')}</div>
        </div>
      )}
    </div>
  );
}
