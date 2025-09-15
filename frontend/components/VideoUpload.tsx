import React, { useRef, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Video, Upload, Trash2, Play, Camera } from 'lucide-react';

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
  onCameraModeChange,
  onOpenCamera
}) => {
  const inputRef = useRef<HTMLInputElement | null>(null);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    onDropRejected: onRejected as any,
    accept: { 'video/mp4': ['.mp4'], 'video/webm': ['.webm'], 'video/avi': ['.avi'] },
    maxSize: MAX_SIZE,
    multiple: false,
    noClick: true, // Disable click to open file dialog
  });

  const handleBrowse = () => inputRef.current?.click();

  return (
    <div className="grid gap-3">
      <div
        {...getRootProps({
          className: `rounded-2xl border-2 border-dashed p-6 transition ${
            isDragReject ? 'border-red-400 bg-red-500/10' : 
            isDragActive || isDragging ? 'border-blue-500 bg-blue-500/5 ring-4 ring-blue-500/30' : 
            'border-white/10'
          }`,
          onDragEnter: () => setIsDragging(true),
          onDragLeave: () => setIsDragging(false),
        })}
      >
        <input {...getInputProps({ 'aria-label': 'Video upload dropzone' })} />
        <div className="grid place-items-center gap-2 text-center min-h-[220px]">
          {!previewUrl ? (
            <>
              <div className="w-14 h-14 grid place-items-center text-blue-500 bg-blue-500/20 rounded-full">
                <Video size={36} />
              </div>
              <p className="font-semibold">Drag & drop video here</p>
              <p className="text-slate-400 -mt-1">MP4, WebM, AVI up to 100MB</p>
              <div className="flex gap-2 mt-3">
                <button 
                  type="button" 
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-blue-500 hover:bg-blue-600 text-white" 
                  onClick={handleBrowse} 
                  aria-label="Browse video files"
                >
                  <Upload size={16} />
                  <span>Browse Videos</span>
                </button>
                <button 
                  type="button" 
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/20 hover:bg-white/10 text-white" 
                  onClick={onOpenCamera} 
                  aria-label="Record video"
                >
                  <Camera size={16} />
                  <span>Record Video</span>
                </button>
              </div>
            </>
          ) : (
            <div className="grid gap-2 w-full">
              <video 
                src={previewUrl} 
                controls 
                className="max-h-96 w-auto max-w-full rounded-xl border border-white/10 bg-slate-950"
                aria-label={`Video preview: ${file?.name}`}
              />
              <div className="flex justify-between text-slate-400 text-sm">
                <p className="truncate max-w-[70%]" title={file?.name}>{file?.name}</p>
                <p>{formatBytes(file?.size || 0)}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="flex gap-2">
        <button 
          type="button" 
          className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-blue-500 hover:bg-blue-600 text-white disabled:opacity-60" 
          onClick={onProcess} 
          disabled={!canProcess} 
          aria-disabled={!canProcess}
        >
          <Play size={16} />
          <span>Process Video</span>
        </button>
        <button 
          type="button" 
          className="inline-flex items-center gap-2 px-4 py-2 rounded-md border border-white/10 hover:bg-white/10 disabled:opacity-60" 
          onClick={onReset} 
          disabled={!file}
        >
          <Trash2 size={16} />
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
    </div>
  );
};

export default VideoUpload;
