import React, { useRef, useEffect, useState, useCallback } from 'react';
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

interface ImageOverlayProps {
  imageSrc: string;
  predictions: Prediction[];
  className?: string;
  onError?: (error: string) => void;
  drawKeypoints?: boolean;
  showBoxes?: boolean;
  boxThreshold?: number; // 0..1
  keypointThreshold?: number; // 0..1
}

export default function ImageOverlay({ imageSrc, predictions, className, onError, drawKeypoints = false, showBoxes = true, boxThreshold = 0, keypointThreshold = 0 }: ImageOverlayProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
  const [isLoaded, setIsLoaded] = useState(false);

    // Draw overlay predictions
  const drawOverlay = useCallback(() => {
    const image = imageRef.current;
    const canvas = canvasRef.current;
    
    if (!image || !canvas || !isLoaded || predictions.length === 0) {
      return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas resolution to match original image dimensions
    canvas.width = imageDimensions.width;
    canvas.height = imageDimensions.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw predictions using original image coordinates (no scaling needed)
    predictions.forEach((pred) => {
      const canDrawBox = !!pred.bbox && showBoxes && (typeof pred.confidence !== 'number' || pred.confidence >= boxThreshold);
      const canDrawKps = drawKeypoints && Array.isArray(pred.keypoints) && pred.keypoints.length > 0;
      if (!pred.bbox && !canDrawKps) return;

      if (canDrawBox && pred.bbox) {
        const { x, y, width, height } = pred.bbox;
  const boxColor = getClassColor(pred.class_id ?? 0);
  // Draw bounding box with per-class color
  ctx.strokeStyle = boxColor;
        ctx.lineWidth = Math.max(2, Math.floor(imageDimensions.height * 0.003)); // Scale line width to image size
        ctx.strokeRect(x, y, width, height);

        // Draw label with background
        const label = `${pred.label || `Class ${pred.class_id}`}: ${pred.confidence.toFixed(2)}`;
        
        // Calculate font size based on original image size
        const fontSize = Math.max(12, Math.floor(imageDimensions.height * 0.02));
        ctx.font = `${fontSize}px Arial`;
        
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = fontSize;

        // Position label above bounding box, or below if too close to top
  const labelY = y > textHeight + 10 ? y - 5 : y + height + textHeight + 5;

  // Draw background rectangle for text in same hue
  ctx.fillStyle = colorWithAlpha(boxColor, 0.9);
        ctx.fillRect(x, labelY - textHeight, textWidth + 8, textHeight + 4);

  // Draw text with suitable contrast
  ctx.fillStyle = getTextColorForBackground(boxColor);
        ctx.fillText(label, x + 4, labelY - 4);
      }

      // Draw keypoints when provided and enabled
      if (canDrawKps) {
        const kpRadius = Math.max(2, Math.floor(imageDimensions.height * 0.006));
        ctx.fillStyle = '#00ffff'; // Cyan for keypoints
        ctx.strokeStyle = '#002233';
        ctx.lineWidth = Math.max(1, Math.floor(imageDimensions.height * 0.002));
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
  }, [predictions, imageDimensions, isLoaded, drawKeypoints, showBoxes, boxThreshold, keypointThreshold]);

  // Handle image load
  const handleImageLoad = useCallback(() => {
    const image = imageRef.current;
    if (!image) return;
    
    setImageDimensions({
      width: image.naturalWidth,
      height: image.naturalHeight
    });
    setIsLoaded(true);
  }, []);

  // Handle image error
  const handleImageError = useCallback(() => {
    const errorMsg = 'Failed to load image';
    console.error(errorMsg);
    onError?.(errorMsg);
  }, [onError]);

  // Redraw overlay when predictions or dimensions change
  useEffect(() => {
    if (isLoaded) {
      drawOverlay();
    }
  }, [drawOverlay, isLoaded]);

  // Handle window resize to redraw overlay with correct dimensions
  useEffect(() => {
    const handleResize = () => {
      if (isLoaded) {
        // Small delay to ensure image has resized
        setTimeout(drawOverlay, 50);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [drawOverlay, isLoaded]);

  return (
    <div className={`relative inline-block ${className || ''}`} style={{ maxWidth: '100%' }}>
      {/* Image element */}
      <img
        ref={imageRef}
        src={imageSrc}
        onLoad={handleImageLoad}
        onError={handleImageError}
        className="block w-auto h-auto max-w-full object-contain"
        style={{ 
          display: 'block',
          maxHeight: '80vh',
          aspectRatio: 'auto'
        }}
        alt="Image with predictions"
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
      />
      
      {/* Debug info */}
      {(typeof (globalThis as any).process !== 'undefined' && (globalThis as any).process?.env?.NODE_ENV === 'development') && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white text-xs p-2 rounded max-w-xs">
          <div>Dimensions: {imageDimensions.width}x{imageDimensions.height}</div>
          <div>Predictions: {predictions.length}</div>
          <div>Loaded: {isLoaded ? 'Yes' : 'No'}</div>
        </div>
      )}
    </div>
  );
}
