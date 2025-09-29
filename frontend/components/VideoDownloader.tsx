import React, { useRef, useState, useCallback } from 'react';
import { Download, Loader2 } from 'lucide-react';
import { getClassColor, colorWithAlpha, getTextColorForBackground } from '../utils/colors';

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
  keypoints?: { x: number; y: number; score?: number }[];
}

interface VideoFrame {
  frame_number: number;
  timestamp: number;
  predictions: Prediction[];
}

interface VideoDownloaderProps {
  videoSrc: string;
  frames: VideoFrame[];
  fileName?: string;
  className?: string;
  showBoxes?: boolean;
  showKeypoints?: boolean;
  boxThreshold?: number; // 0..1
  keypointThreshold?: number; // 0..1
}

const VideoDownloader: React.FC<VideoDownloaderProps> = ({ 
  videoSrc, 
  frames, 
  fileName = 'processed_video.mp4',
  className = '',
  showBoxes = true,
  showKeypoints = true,
  boxThreshold = 0,
  keypointThreshold = 0,
}) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Create lookup for frame predictions by timestamp
  const predictionLookup = useCallback(() => {
    const lookup = new Map<number, Prediction[]>();
    frames.forEach(frame => {
      if (frame.predictions && frame.predictions.length > 0) {
        const roundedTime = Math.round(frame.timestamp * 10) / 10;
        lookup.set(roundedTime, frame.predictions);
      }
    });
    return lookup;
  }, [frames]);

  const drawBoundingBoxes = useCallback((ctx: CanvasRenderingContext2D, predictions: Prediction[], videoWidth: number, videoHeight: number) => {
    predictions.forEach((pred) => {
      // Draw bbox when present and allowed
      const canDrawBox = !!pred.bbox && showBoxes && (typeof pred.confidence !== 'number' || pred.confidence >= boxThreshold);
      if (canDrawBox && pred.bbox) {
        const { x, y, width, height } = pred.bbox;
        const boxColor = getClassColor(pred.class_id ?? 0);
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        // Draw label with background
        const label = `${pred.label || `Class ${pred.class_id}`}: ${pred.confidence.toFixed(2)}`;
        const fontSize = Math.floor(videoHeight * 0.02);
        ctx.font = `${fontSize}px Arial`;
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;
        const labelY = y > textHeight + 10 ? y - 5 : y + height + textHeight + 5;
        ctx.fillStyle = colorWithAlpha(boxColor, 0.9);
        ctx.fillRect(x, labelY - textHeight, textWidth + 8, textHeight + 4);
        ctx.fillStyle = getTextColorForBackground(boxColor);
        ctx.fillText(label, x + 4, labelY - 4);
      }

      // Draw keypoints when present and allowed
      if (showKeypoints && Array.isArray(pred.keypoints) && pred.keypoints.length > 0) {
        const kpRadius = Math.max(2, Math.floor(videoHeight * 0.012));
        ctx.fillStyle = '#00ffff';
        ctx.strokeStyle = '#002233';
        ctx.lineWidth = Math.max(1, Math.floor(videoHeight * 0.004));
        for (const kp of pred.keypoints) {
          if (typeof kp.x === 'number' && typeof kp.y === 'number') {
            if (typeof kp.score === 'number' && kp.score < keypointThreshold) continue;
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, kpRadius, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }
      }
    });
  }, [showBoxes, showKeypoints, boxThreshold, keypointThreshold]);

  const processVideoFrame = useCallback(async (
    video: HTMLVideoElement, 
    canvas: HTMLCanvasElement, 
    ctx: CanvasRenderingContext2D, 
    predictions: Map<number, Prediction[]>,
    currentTime: number
  ): Promise<ImageData> => {
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame
    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);

    // Find predictions for current time
    const roundedTime = Math.round(currentTime * 10) / 10;
    let currentPredictions: Prediction[] = [];

    // Look for predictions within a small time window
    for (let offset = -0.2; offset <= 0.2; offset += 0.1) {
      const searchTime = Math.round((roundedTime + offset) * 10) / 10;
      if (predictions.has(searchTime)) {
        currentPredictions = predictions.get(searchTime) || [];
        break;
      }
    }

    // Draw bounding boxes if found
    if (currentPredictions.length > 0) {
      drawBoundingBoxes(ctx, currentPredictions, video.videoWidth, video.videoHeight);
    }

    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  }, [drawBoundingBoxes]);

  const downloadVideo = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    
    if (!video || !canvas) {
      console.error('Video or canvas not available');
      return;
    }

    setIsProcessing(true);
    setProgress(0);

    try {
      const predictions = predictionLookup();
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Could not get canvas context');

      // Wait for video to load
      await new Promise<void>((resolve, reject) => {
        if (video.readyState >= 2) {
          resolve();
        } else {
          video.addEventListener('loadeddata', () => resolve(), { once: true });
          video.addEventListener('error', reject, { once: true });
        }
      });

      const stream = canvas.captureStream(30); // 30 FPS
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm; codecs=vp9'
      });

      const chunks: Blob[] = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName.replace('.mp4', '.webm'); // MediaRecorder outputs WebM
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        setIsProcessing(false);
        setProgress(100);
      };

      mediaRecorder.start();

      // Process video by seeking through timestamps
      const duration = video.duration;
      const frameRate = 30; // Target frame rate
      const frameInterval = 1 / frameRate;
      let currentTime = 0;

      const processNextFrame = async () => {
        if (currentTime >= duration) {
          mediaRecorder.stop();
          return;
        }

        video.currentTime = currentTime;
        
        await new Promise<void>((resolve) => {
          const onSeeked = () => {
            video.removeEventListener('seeked', onSeeked);
            resolve();
          };
          video.addEventListener('seeked', onSeeked);
        });

        await processVideoFrame(video, canvas, ctx, predictions, currentTime);
        
        setProgress((currentTime / duration) * 100);
        currentTime += frameInterval;
        
        // Use requestAnimationFrame for smooth processing
        requestAnimationFrame(processNextFrame);
      };

      await processNextFrame();

    } catch (error) {
      console.error('Error processing video:', error);
      setIsProcessing(false);
      setProgress(0);
    }
  }, [videoSrc, frames, fileName, predictionLookup, processVideoFrame]);

  return (
    <div className={className}>
      {/* Hidden video element for processing */}
      <video
        ref={videoRef}
        src={videoSrc}
        className="hidden"
        preload="metadata"
        muted
      />
      
      {/* Hidden canvas for rendering */}
      <canvas ref={canvasRef} className="hidden" />
      
      {/* Download button */}
      <button
        onClick={downloadVideo}
        disabled={isProcessing || frames.length === 0}
        className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:opacity-50 text-white text-sm transition-colors"
        title="Download video with bounding boxes"
      >
        {isProcessing ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Processing {progress.toFixed(0)}%</span>
          </>
        ) : (
          <>
            <Download className="w-4 h-4" />
            <span>Download Video</span>
          </>
        )}
      </button>
    </div>
  );
};

export default VideoDownloader;
