import React, { useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { ImagePlus, Upload, Trash2, Eye, EyeOff, Camera } from 'lucide-react';

const MAX_SIZE = 10 * 1024 * 1024; // 10MB

function formatBytes(bytes: number) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

type Props = {
  onDrop: (files: File[]) => void;
  onRejected: (rejections: any[]) => void;
  isDragging: boolean;
  setIsDragging: (val: boolean) => void;
  file: File | null;
  previewUrl: string;
  processedImageUrl?: string;
  showProcessedImage?: boolean;
  onToggleImage?: () => void;
  onPredict: () => void;
  canPredict: boolean;
  onReset: () => void;
  onCameraModeChange?: (mode: 'image' | 'video') => void;
  onOpenCamera?: () => void;
};

const ImageUpload: React.FC<Props> = ({ 
  onDrop, 
  onRejected, 
  isDragging, 
  setIsDragging, 
  file, 
  previewUrl, 
  processedImageUrl,
  showProcessedImage,
  onToggleImage,
  onPredict, 
  canPredict, 
  onReset,
  onCameraModeChange,
  onOpenCamera
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    onDropRejected: onRejected,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    maxSize: MAX_SIZE,
    multiple: false,
    noClick: true,
  } as const);

  const handleBrowse = () => inputRef.current?.click();

  // Determine which image to show
  const currentImageUrl = showProcessedImage && processedImageUrl ? processedImageUrl : previewUrl;
  const hasProcessedImage = !!processedImageUrl;

  return (
    <div className="grid gap-3">

      <div
        {...getRootProps({
          className: `rounded-2xl border-2 border-dashed p-6 transition ${
            isDragReject ? 'border-red-400 bg-red-500/10' : isDragActive || isDragging ? 'border-brand-500 bg-brand-500/5 ring-4 ring-brand-500/30' : 'border-white/10'
          }`,
          onDragEnter: () => setIsDragging(true),
          onDragLeave: () => setIsDragging(false),
        })}
      >
        <input {...getInputProps({ 'aria-label': 'Image upload dropzone' })} />
        <div className="grid place-items-center gap-2 text-center min-h-[220px]">
          {!currentImageUrl ? (
            <>
              <div className="w-14 h-14 grid place-items-center text-brand-500 bg-brand-500/20 rounded-full">
                <ImagePlus size={36} />
              </div>
              <p className="font-semibold">Drag & drop image here</p>
              <p className="text-slate-400 -mt-1">PNG, JPG up to 10MB</p>
              <div className="flex gap-2 mt-3">
                <button 
                  type="button" 
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-brand-500 hover:bg-brand-600 text-white" 
                  onClick={handleBrowse} 
                  aria-label="Browse files"
                >
                  <Upload size={16} />
                  <span>Browse Files</span>
                </button>
                <button 
                  type="button" 
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/20 hover:bg-white/10 text-white" 
                  onClick={onOpenCamera} 
                  aria-label="Take photo"
                >
                  <Camera size={16} />
                  <span>Take Photo</span>
                </button>
              </div>
            </>
          ) : (
            <div className="grid gap-2 w-full">
              <div className="relative">
                <img src={currentImageUrl} alt={file?.name || 'Selected image preview'} className="max-h-96 w-auto max-w-full rounded-xl border border-white/10 object-contain bg-slate-950" />
                {hasProcessedImage && (
                  <div className="absolute top-2 right-2">
                    <div className="px-2 py-1 bg-black/60 rounded-md text-xs text-white">
                      {showProcessedImage ? 'Processed' : 'Original'}
                    </div>
                  </div>
                )}
              </div>
              <div className="flex justify-between text-slate-400 text-sm">
                <p className="truncate max-w-[70%]" title={file?.name}>{file?.name}</p>
                <p>{formatBytes(file?.size || 0)}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <button type="button" className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-brand-500 hover:bg-brand-600 text-white disabled:opacity-60" onClick={onPredict} disabled={!canPredict} aria-disabled={!canPredict}>
          <Upload size={16} />
          <span>Predict</span>
        </button>
        {hasProcessedImage && onToggleImage && (
          <button type="button" className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10" onClick={onToggleImage}>
            {showProcessedImage ? <EyeOff size={16} /> : <Eye size={16} />}
            <span>{showProcessedImage ? 'Show Original' : 'Show Processed'}</span>
          </button>
        )}
        
        <button type="button" className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 disabled:opacity-60" onClick={onReset} disabled={!file}>
          <Trash2 size={16} />
          <span>Clear</span>
        </button>
      </div>

      <input ref={inputRef} type="file" accept="image/png, image/jpeg" className="hidden" onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])} />
    </div>
  );
};

export default ImageUpload;
