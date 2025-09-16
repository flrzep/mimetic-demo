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

interface ImageOverlayProps {
  imageSrc: string;
  predictions: Prediction[];
  className?: string;
  onError?: (error: string) => void;
}

export default function ImageOverlay({ imageSrc, predictions, className, onError }: ImageOverlayProps) {
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
      if (!pred.bbox) return;
      
      const { x, y, width, height } = pred.bbox;

      // Draw bounding box with bright color for visibility
      ctx.strokeStyle = '#00ff00'; // Bright green
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

      // Draw background rectangle for text
      ctx.fillStyle = 'rgba(0, 255, 0, 0.9)';
      ctx.fillRect(x, labelY - textHeight, textWidth + 8, textHeight + 4);

      // Draw text
      ctx.fillStyle = '#000000';
      ctx.fillText(label, x + 4, labelY - 4);
    });
  }, [predictions, imageDimensions, isLoaded]);

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
      {process.env.NODE_ENV === 'development' && (
        <div className="absolute bottom-2 left-2 bg-black bg-opacity-75 text-white text-xs p-2 rounded max-w-xs">
          <div>Dimensions: {imageDimensions.width}x{imageDimensions.height}</div>
          <div>Predictions: {predictions.length}</div>
          <div>Loaded: {isLoaded ? 'Yes' : 'No'}</div>
        </div>
      )}
    </div>
  );
}
