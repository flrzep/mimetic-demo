import React, { useRef, useCallback } from 'react';
import { useDropzone, DropzoneOptions } from 'react-dropzone';
import { Video, Upload, Trash2, Play, Camera } from 'lucide-react';
import { isMobileDevice } from '../utils/device';

const MAX_SIZE = 100 * 1024 * 1024; // 100MB for videos

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
  onProcess: () => void;
  canProcess: boolean;
  onReset: () => void;
  onCameraModeChange?: (mode: 'image' | 'video') => void;
  onOpenCamera?: () => void;
};

const VideoUpload: React.FC<Props> = ({ 
  onDrop, 
  onRejected, 
  isDragging, 
  setIsDragging, 
  file, 
  previewUrl, 
  onProcess, 
  canProcess, 
  onReset,
  onOpenCamera 
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cameraInputRef = useRef<HTMLInputElement | null>(null);

  const dropzoneOptions: DropzoneOptions = {
    onDrop,
    onDropRejected: onRejected,
    accept: { 'video/mp4': ['.mp4'], 'video/webm': ['.webm'], 'video/avi': ['.avi'] },
    maxSize: MAX_SIZE,
    multiple: false,
    noClick: true,
  };

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone(dropzoneOptions);

  const handleBrowse = () => inputRef.current?.click();
  
  const handleRecordVideo = () => {
    // On mobile devices, this will open the native camera app for video recording
    cameraInputRef.current?.click();
  };

  const isOnMobile = isMobileDevice();

  return (
    <div className="grid gap-3">
      <div
        {...getRootProps({
          className: `rounded-2xl border-2 border-dashed p-4 sm:p-6 transition ${
            isDragReject ? 'border-red-400 bg-red-500/10' : 
            isDragActive || isDragging ? 'border-blue-500 bg-blue-500/5 ring-4 ring-blue-500/30' : 
            'border-white/10'
          }`,
          onDragEnter: () => setIsDragging(true),
          onDragLeave: () => setIsDragging(false),
        })}
      >
        <input {...getInputProps({ 'aria-label': 'Video upload dropzone' })} />
        <div className="grid place-items-center gap-2 text-center min-h-[180px] sm:min-h-[220px]">
          {!previewUrl ? (
            <>
              <div className="w-12 h-12 sm:w-14 sm:h-14 grid place-items-center text-blue-500 bg-blue-500/20 rounded-full">
                <Video className="w-7 h-7 sm:w-9 sm:h-9" />
              </div>
              <p className="font-semibold text-sm sm:text-base">Drag & drop video here</p>
              <p className="text-slate-400 -mt-1 text-xs sm:text-sm">MP4, WebM, AVI up to 100MB</p>
              <div className="flex flex-col sm:flex-row gap-2 mt-3 w-full sm:w-auto">
                <button 
                  type="button" 
                  className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-blue-500 hover:bg-blue-600 text-white text-sm sm:text-base" 
                  onClick={handleBrowse} 
                  aria-label="Browse video files"
                >
                  <Upload className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>Browse Videos</span>
                </button>
                <button 
                  type="button" 
                  className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/20 hover:bg-white/10 text-white text-sm sm:text-base" 
                  onClick={handleRecordVideo} 
                  aria-label="Record video with camera"
                >
                  <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                  <span>{isOnMobile ? 'Camera' : 'Record Video'}</span>
                </button>
              </div>
            </>
          ) : (
            <div className="grid gap-2 w-full">
              <video 
                src={previewUrl} 
                controls 
                playsInline
                {...({ 'webkit-playsinline': 'true' } as any)}
                {...({ 'x-webkit-airplay': 'allow' } as any)}
                controlsList="nodownload nofullscreen noremoteplayback"
                disablePictureInPicture
                className="max-h-72 sm:max-h-96 w-auto max-w-full mx-auto rounded-xl border border-white/10 bg-slate-950"
                aria-label={`Video preview: ${file?.name}`}
                preload="metadata"
              />
              <div className="flex justify-between text-slate-400 text-xs sm:text-sm">
                <p className="truncate max-w-[60%] sm:max-w-[70%]" title={file?.name}>{file?.name}</p>
                <p>{formatBytes(file?.size || 0)}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-2">
        <button 
          type="button" 
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-60 text-sm sm:text-base" 
          onClick={onProcess} 
          disabled={!canProcess} 
          aria-disabled={!canProcess}
        >
          <Play className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span>Process Video</span>
        </button>
        <button 
          type="button" 
          className="inline-flex items-center justify-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 disabled:opacity-60 text-sm sm:text-base" 
          onClick={onReset} 
          disabled={!file}
        >
          <Trash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
          <span>Clear</span>
        </button>
      </div>

      <input 
        ref={inputRef} 
        type="file" 
        accept="video/mp4,video/webm,video/avi" 
        className="hidden" 
        onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])} 
      />
      
      {/* Camera input for mobile video recording */}
      <input 
        ref={cameraInputRef} 
        type="file" 
        accept="video/*" 
        capture="environment"
        className="hidden" 
        onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])} 
      />
    </div>
  );
};

export default VideoUpload;
