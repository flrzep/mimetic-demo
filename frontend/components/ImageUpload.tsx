import React, { useRef, useCallback } from 'react';
import { useDropzone, DropzoneOptions } from 'react-dropzone';
import { ImagePlus, Upload, Trash2, Camera } from 'lucide-react';
import { isMobileDevice } from '../utils/device';

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
  onPredict, 
  canPredict, 
  onReset,
  onCameraModeChange,
  onOpenCamera
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);

  const dropzoneOptions: DropzoneOptions = {
    onDrop,
    onDropRejected: onRejected,
    accept: { 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'] },
    maxSize: MAX_SIZE,
    multiple: false,
    noClick: true,
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone(dropzoneOptions);

  const handleBrowse = () => inputRef.current?.click();
  
  const handleTakePhoto = () => {
    // Use the camera modal instead of native capture for better UX
    if (onOpenCamera) {
      onOpenCamera();
    } else {
      // Fallback to native camera input if modal not available
      cameraInputRef.current?.click();
    }
  };

  const isOnMobile = isMobileDevice();

  return (
    <div className="grid gap-3">

      <div
        {...getRootProps({
          className: `rounded-2xl border-2 border-dashed p-4 sm:p-6 transition ${
            isDragReject ? 'border-red-400 bg-red-500/10' : isDragActive || isDragging ? 'border-brand-500 bg-brand-500/5 ring-4 ring-brand-500/30' : 'border-white/10'
          }`,
          onDragEnter: () => setIsDragging(true),
          onDragLeave: () => setIsDragging(false),
        })}
      >
        <input {...getInputProps({ 'aria-label': 'Image upload dropzone' })} />
        <div className="grid place-items-center gap-2 text-center min-h-[180px] sm:min-h-[220px]">
          {!previewUrl ? (
            <>
              <div className="w-12 h-12 sm:w-14 sm:h-14 grid place-items-center text-brand-500 bg-brand-500/20 rounded-full">
                <ImagePlus className="w-7 h-7 sm:w-9 sm:h-9" />
              </div>
              <p className="font-semibold text-sm sm:text-base">Drag & drop image here</p>
              <p className="text-slate-400 -mt-1 text-xs sm:text-sm">PNG, JPG up to 10MB</p>
              <div className="flex flex-col sm:flex-row gap-2 mt-3 w-full sm:w-auto">
                <button 
                  type="button" 
                  className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-brand-500 hover:bg-brand-600 text-white text-sm sm:text-base" 
                  onClick={handleBrowse} 
                  aria-label="Browse files"
                >
                  <Upload className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>Browse Files</span>
                </button>
                <button 
                  type="button" 
                  className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/20 hover:bg-white/10 text-white text-sm sm:text-base"
                  onClick={handleTakePhoto} 
                  aria-label="Take photo with camera"
                >
                  <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>{isOnMobile ? 'Camera' : 'Take Photo'}</span>
                </button>
              </div>
            </>
          ) : (
            <div className="grid gap-2 w-full">
              <div className="relative">
                <img src={previewUrl} alt={file?.name || 'Selected image preview'} className="max-h-72 sm:max-h-96 w-auto max-w-full mx-auto rounded-xl border border-white/10 object-contain bg-slate-950" />
              </div>
              <div className="flex justify-between text-slate-400 text-xs sm:text-sm">
                <p className="truncate max-w-[60%] sm:max-w-[70%]" title={file?.name}>{file?.name}</p>
                <p>{formatBytes(file?.size || 0)}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-2">
        <button type="button" className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-brand-500 hover:bg-brand-600 text-white disabled:opacity-60 text-sm sm:text-base" onClick={onPredict} disabled={!canPredict} aria-disabled={!canPredict}>
          <Upload className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span>Predict</span>
        </button>
        
        <button type="button" className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 disabled:opacity-60 text-sm sm:text-base" onClick={onReset} disabled={!file}>
          <Trash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span>Clear</span>
        </button>
      </div>

      <input ref={inputRef} type="file" accept="image/png, image/jpeg" className="hidden" onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])} />
      
      {/* Camera input for mobile photo capture */}
      <input 
        ref={cameraInputRef} 
        type="file" 
        accept="image/*" 
        capture="environment"
        className="hidden" 
        onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])} 
      />
    </div>
  );
};

export default ImageUpload;
