"use client";

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import Image from 'next/image';
import { AlertTriangle, Timer, Image as ImageIcon, Video, Radio, ChevronDown, Brain } from 'lucide-react';
import { ImageUpload, VideoUpload, StreamProcessor, PredictionResults, LoadingSpinner, ServerStatus, VideoFrameResults, VideoDownloader } from '../components';
import CameraModal from '../components/CameraModal';
import VideoOverlay from '../components/VideoOverlay';
import ImageOverlay from '../components/ImageOverlay';
import { checkHealth, predictImage, predictVideo } from '../utils/api';
import { supabase } from '../lib/supabase';

const REQUIRE_AUTH = process.env.NEXT_PUBLIC_REQUIRE_AUTH === 'true';

type Prediction = { class_id: number; confidence: number; label?: string };
type Results = { 
  predictions: Prediction[]; 
  processing_time?: number;
  total_frames?: number;
  processed_frames?: number;
} | null;
type TabType = 'image' | 'video' | 'stream';

type ModelOption = {
  id: string;
  name: string;
  description: string;
  category: 'classification' | 'detection' | 'segmentation' | 'generation';
  recommended?: boolean;
};

const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'resnet50',
    name: 'ResNet-50',
    description: 'General image classification, fast and reliable',
    category: 'classification',
    recommended: true
  },
  {
    id: 'efficientnet_b4',
    name: 'EfficientNet-B4',
    description: 'High accuracy image classification with efficiency',
    category: 'classification'
  },
  {
    id: 'yolov8',
    name: 'YOLOv8',
    description: 'Real-time object detection and localization',
    category: 'detection'
  },
  {
    id: 'detr',
    name: 'DETR',
    description: 'Transformer-based object detection',
    category: 'detection'
  },
  {
    id: 'sam',
    name: 'Segment Anything (SAM)',
    description: 'Universal image segmentation model',
    category: 'segmentation'
  },
  {
    id: 'stable_diffusion',
    name: 'Stable Diffusion',
    description: 'AI image generation and editing',
    category: 'generation'
  }
];

export default function Page() {
  const [activeTab, setActiveTab] = useState<TabType>('image');
  const [selectedModel, setSelectedModel] = useState<string>('resnet50');
  const [isModelDropdownOpen, setIsModelDropdownOpen] = useState(false);
  
  // Camera modal state - moved to main page to persist across tab switches
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraMode, setCameraMode] = useState<'image' | 'video'>('image');

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (!target.closest('[data-model-dropdown]')) {
        setIsModelDropdownOpen(false);
      }
    };

    if (isModelDropdownOpen) {
      document.addEventListener('click', handleClickOutside);
      return () => document.removeEventListener('click', handleClickOutside);
    }
  }, [isModelDropdownOpen]);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<Results>(null);
  const [error, setError] = useState('');
  const [health, setHealth] = useState<{ ok: boolean; message: string }>({ ok: true, message: '' });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [latency, setLatency] = useState<number | undefined>(undefined);
  const [user, setUser] = useState<any>(null);
  
  // Video processing state
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState('');
  const [processedVideoUrl, setProcessedVideoUrl] = useState('');
  const [videoFrames, setVideoFrames] = useState<any[]>([]);  // Store frame predictions for overlay
  const [currentVideoTime, setCurrentVideoTime] = useState<number>(0);  // Track current video time
  const [isVideoProcessing, setIsVideoProcessing] = useState(false);
  const [videoPlaybackError, setVideoPlaybackError] = useState(false);
  const [hideProcessedVideo, setHideProcessedVideo] = useState(false);
  
  // Stream processing state
  const [isStreaming, setIsStreaming] = useState(false);

  // Reset function for tab changes
  const resetAllStates = () => {
    // Clean up blob URLs before resetting
    if (processedVideoUrl && processedVideoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(processedVideoUrl);
    }
    
    setFile(null);
    setPreviewUrl('');
    setVideoFile(null);
    setVideoPreviewUrl('');
    setProcessedVideoUrl('');
    setResults(null);
    setError('');
    setIsLoading(false);
    setIsVideoProcessing(false);
    setUploadProgress(0);
  };

  useEffect(() => {
    if (!REQUIRE_AUTH || !supabase) return;
    let mounted = true;
    supabase.auth.getUser().then(({ data }) => mounted && setUser(data?.user || null));
    const { data: sub } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user || null);
    });
    return () => { mounted = false; (sub as any)?.subscription?.unsubscribe?.(); };
  }, []);

  const signInWithEmail = async () => {
    if (!supabase) {
      setError('Authentication not configured');
      return;
    }
    const email = window.prompt('Enter email for magic link:');
    if (!email) return;
    const { error } = await supabase.auth.signInWithOtp({ email, options: { emailRedirectTo: window.location.origin } });
    if (error) setError(error.message);
  };

  const signOut = async () => { 
    if (supabase) await supabase.auth.signOut(); 
  };

  useEffect(() => {
    let isMounted = true;
    const ping = async () => {
      try {
        const start = performance.now();
        const status = await checkHealth();
        if (!isMounted) return;
        setHealth(status);
        setLatency(performance.now() - start);
      } catch (e) {
        if (!isMounted) return;
        setHealth({ ok: false, message: 'Backend unreachable' });
        setLatency(undefined);
      }
    };
    ping();
    const t = setInterval(ping, 15000);
    return () => { isMounted = false; clearInterval(t); };
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!videoFile) return;
    const url = URL.createObjectURL(videoFile);
    setVideoPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [videoFile]);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      if (processedVideoUrl && processedVideoUrl.startsWith('blob:')) {
        URL.revokeObjectURL(processedVideoUrl);
      }
    };
  }, [processedVideoUrl]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles?.length) return;
    setError('');
    setResults(null);
    setFile(acceptedFiles[0]);
  }, []);

  const onVideoDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles?.length) return;
    setError('');
    setResults(null);
    setVideoFile(acceptedFiles[0]);
  }, []);

  const onRejected = useCallback((fileRejections: any[]) => {
    if (!fileRejections?.length) return;
    const first = fileRejections[0];
    const reasons = first.errors?.map((e: any) => e.message).join(', ');
    setError(reasons || 'File rejected');
  }, []);

  const reset = useCallback(() => {
    setResults(null);
    setError('');
    setFile(null);
    setPreviewUrl('');
  }, []);

  const canPredict = useMemo(() => !!file && !isLoading, [file, isLoading]);

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const predict = useCallback(async () => {
    if (!file) { setError('Please select an image first.'); return; }
    setIsLoading(true);
    setError('');
    setResults(null);
    setUploadProgress(0);
    try {
      const start = performance.now();
      const maxAttempts = 3;
      let lastErr: any = null;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          const res = await predictImage(file, (e: ProgressEvent) => {
            const total = (e as any).total;
            if (total) setUploadProgress(Math.round(((e as any).loaded / total) * 100));
          });
          const elapsed = (performance.now() - start) / 1000;
          const data = (res as any).data;
          if (!data?.success) throw new Error(data?.message || 'Prediction failed');
          
          setResults({ predictions: data.predictions || [], processing_time: data.processing_time ?? elapsed });
          lastErr = null; break;
        } catch (e: any) {
          lastErr = e;
          const retriable = e.code === 'ECONNABORTED' || e.message?.toLowerCase().includes('network') || [502,503,504].includes(e.response?.status);
          if (attempt < maxAttempts && retriable) {
            const delay = 1000 * Math.pow(2, attempt - 1) + Math.random() * 500;
            await sleep(delay); continue;
          }
          throw e;
        }
      }
      if (lastErr) throw lastErr;
    } catch (e: any) {
      const msg = e.code === 'ECONNABORTED'
        ? 'Request timed out. The backend may be waking up (Render cold start). Please try again in a few seconds.'
        : e.response?.data?.message || e.message || 'An error occurred during prediction.';
      setError(msg);
    } finally { setIsLoading(false); }
  }, [file]);

  // Video processing handlers
  const canProcessVideo = useMemo(() => !!videoFile && !isVideoProcessing, [videoFile, isVideoProcessing]);

  const processVideo = useCallback(async () => {
    if (!videoFile) { setError('Please select a video first.'); return; }
    setIsVideoProcessing(true);
    setError('');
    setResults(null);
    setUploadProgress(0);
    setVideoPlaybackError(false); // Reset video error state
    setHideProcessedVideo(false); // Reset video hide state
    try {
      const start = performance.now();
      
      // Configure video processing options based on the recorded video
      // Detect the format from the video file type or use a reasonable default
      let detectedFormat = 'mp4'; // Default fallback
      
      if (videoFile.type) {
        if (videoFile.type.includes('webm')) {
          detectedFormat = 'webm';
        } else if (videoFile.type.includes('mp4')) {
          detectedFormat = 'mp4';
        }
      } else if (videoFile.name) {
        // Fallback to filename extension
        const extension = videoFile.name.split('.').pop()?.toLowerCase();
        if (extension === 'webm') {
          detectedFormat = 'webm';
        } else if (extension === 'mp4' || extension === 'm4v') {
          detectedFormat = 'mp4';
        }
      }
      
      console.log('Video file details:', {
        name: videoFile.name,
        type: videoFile.type,
        size: videoFile.size,
        detectedFormat
      });
      
      const options = {
        frame_interval: 1, // Process every frame
        max_frames: 300,   // Limit to 300 frames for demo
        output_format: detectedFormat, // Use detected format to preserve original
        video_codec: detectedFormat === 'webm' ? 'vp8' : 'h264', // Correct codec mapping
        audio_codec: 'none',  // No audio track
        return_url: false     // Use base64 for better compatibility
      };
      
      const res = await predictVideo(videoFile, options, (e: ProgressEvent) => {
        const total = (e as any).total;
        if (total) setUploadProgress(Math.round(((e as any).loaded / total) * 100));
      });
      
      const elapsed = (performance.now() - start) / 1000;
      const data = (res as any).data;
      
      if (!data) throw new Error('No response data received');
      
      // Store the processed video if available
      // For client-side overlay, we use the original video file directly
      // No need to process video data from backend
      console.log('Using original video file for client-side overlay');
      
      // Create blob URL from the original video file if we don't have one
      if (!videoPreviewUrl && videoFile) {
        const originalVideoUrl = URL.createObjectURL(videoFile);
        setVideoPreviewUrl(originalVideoUrl);
        console.log('Created blob URL from original video file:', originalVideoUrl);
      }
      
      // Use the original video preview URL for the overlay
      setProcessedVideoUrl(videoPreviewUrl || URL.createObjectURL(videoFile));
      
      // Store frame predictions for client-side overlay
      const framePredictions = data.frames || [];
      setVideoFrames(framePredictions);
      
      // Debug frame predictions
      console.log('📹 Frame predictions received:', {
        totalFrames: framePredictions.length,
        sampleFrame: framePredictions[0],
        framesWithPredictions: framePredictions.filter((f: any) => f.predictions && f.predictions.length > 0).length,
        framesWithBboxes: framePredictions.filter((f: any) => 
          f.predictions && f.predictions.some((p: any) => p.bbox)
        ).length
      });
      
      // Log sample predictions
      if (framePredictions.length > 0) {
        const sampleFrame = framePredictions[0];
        console.log('🎯 Sample frame predictions:', {
          frameNumber: sampleFrame.frame_number,
          timestamp: sampleFrame.timestamp,
          predictions: sampleFrame.predictions?.map((p: any) => ({
            label: p.label,
            confidence: p.confidence,
            hasBbox: !!p.bbox,
            bbox: p.bbox
          }))
        });
      }
      
      // Extract frame predictions for results display
      const allPredictions = framePredictions.flatMap((frame: any) => 
        frame.predictions.map((pred: any) => ({
          ...pred,
          frame_number: frame.frame_number,
          timestamp: frame.timestamp
        }))
      );
      
      setResults({ 
        predictions: allPredictions, 
        processing_time: data.processing_time ?? elapsed,
        total_frames: data.total_frames,
        processed_frames: data.processed_frames
      });
      
    } catch (e: any) {
      setError(e.message || 'Video processing failed');
    } finally { 
      setIsVideoProcessing(false); 
    }
  }, [videoFile]);

  const resetVideo = useCallback(() => {
    // Clean up blob URLs before resetting
    if (processedVideoUrl && processedVideoUrl.startsWith('blob:')) {
      URL.revokeObjectURL(processedVideoUrl);
    }
    
    setVideoFile(null);
    setVideoPreviewUrl('');
    setProcessedVideoUrl('');
    setVideoFrames([]);
    setVideoPlaybackError(false);
    setHideProcessedVideo(false);
    setResults(null);
    setError('');
  }, [processedVideoUrl]);

  // Stream processing handlers
  const startStreamProcessing = useCallback(() => {
    setIsStreaming(true);
    setError('');
    // Placeholder for stream processing start
  }, []);

  const stopStreamProcessing = useCallback(() => {
    setIsStreaming(false);
    setResults(null);
  }, []);

  const handleStreamSettings = useCallback(() => {
    // Placeholder for stream settings
    alert('Stream settings coming soon!');
  }, []);

  // Tab change handler
  const handleTabChange = useCallback((tab: TabType) => {
    setActiveTab(tab);
    resetAllStates();
  }, []);

  // Camera mode change handler
  const handleCameraModeChange = useCallback((mode: 'image' | 'video') => {
    setCameraMode(mode); // Update camera mode
    if (mode === 'image') {
      setActiveTab('image');
    } else if (mode === 'video') {
      setActiveTab('video');
    }
    // Don't reset states when switching from camera modal to keep it open
  }, []);

  // Camera capture handlers
  const handleCameraCapture = useCallback((capturedFile: File) => {
    console.log('Original captured file (this works):', {
      name: capturedFile.name,
      size: capturedFile.size,
      type: capturedFile.type,
      lastModified: capturedFile.lastModified
    });
    
    if (cameraMode === 'image') {
      setFile(capturedFile);
      setPreviewUrl(URL.createObjectURL(capturedFile));
    } else {
      setVideoFile(capturedFile);
      setVideoPreviewUrl(URL.createObjectURL(capturedFile));
    }
    setIsCameraOpen(false);
  }, [cameraMode]);

  const openCamera = useCallback((mode: 'image' | 'video') => {
    setCameraMode(mode);
    setIsCameraOpen(true);
  }, []);

  return (
    <div className="min-h-screen text-slate-200 bg-[linear-gradient(180deg,#0f172a,#0b1020_60%)]">
      <header className="py-4 sm:py-8 mb-6">
        <div className="max-w-3xl mx-auto px-4">
          <div className="flex flex-row sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3 sm:gap-4">
              <Image
                src="/logo-title.png"
                alt="Computer Vision Demo Logo"
                width={180}
                height={50}
                className="h-8 sm:h-10 w-auto"
                priority
              />
              <div>
                <h1 className="m-0 text-xl sm:text-2xl font-bold tracking-tight">Demo</h1>
                <p className="mt-0 text-sm sm:text-base text-slate-400">Test our models online</p>
              </div>
            </div>
            
            {/* Sign In/Out Button */}
            {REQUIRE_AUTH && (
              <div className="self-end sm:self-auto">
                {user ? (
                  <button className="px-3 py-1.5 rounded-md border border-white/10 hover:bg-white/10 text-sm" onClick={signOut} aria-label="Sign out">Sign out</button>
                ) : (
                  <button className="px-3 py-1.5 rounded-md bg-brand-500 hover:bg-brand-600 text-white text-sm" onClick={signInWithEmail} aria-label="Sign in">Sign in</button>
                )}
              </div>
            )}
          </div>
        </div>
      </header>

      <main className="max-w-3xl mx-auto px-3 sm:px-4 grid gap-3 sm:gap-4 pb-8">
        {/* Model Selector and Server Status Row */}
        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3 sm:gap-4">
          <div className="flex-1">
            <div className="relative" data-model-dropdown>
              <button
                onClick={() => setIsModelDropdownOpen(!isModelDropdownOpen)}
                className="flex items-center gap-2 px-3 sm:px-4 py-2 rounded-lg border border-white/10 bg-slate-900/50 hover:bg-slate-800/50 transition-colors w-full lg:min-w-[280px]"
                aria-expanded={isModelDropdownOpen}
                aria-haspopup="listbox"
              >
                <Brain className="w-4 h-4 sm:w-[18px] sm:h-[18px] text-brand-500 flex-shrink-0" />
                <div className="flex-1 text-left min-w-0">
                  <div className="font-medium text-sm sm:text-base truncate">
                    {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.name}
                    {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.recommended && (
                      <span className="ml-2 px-1.5 py-0.5 text-xs bg-green-500/20 text-green-300 rounded">Recommended</span>
                    )}
                  </div>
                  <div className="text-xs sm:text-sm text-slate-400 truncate">
                    {AVAILABLE_MODELS.find(m => m.id === selectedModel)?.description}
                  </div>
                </div>
                <ChevronDown className="w-3.5 h-3.5 sm:w-4 sm:h-4 transition-transform flex-shrink-0" style={{ transform: isModelDropdownOpen ? 'rotate(180deg)' : 'rotate(0deg)' }} />
              </button>

              {isModelDropdownOpen && (
                <div className="absolute top-full left-0 right-0 mt-1 py-1 bg-slate-900 border border-white/10 rounded-lg shadow-xl z-50 max-h-80 overflow-y-auto">
                  {AVAILABLE_MODELS.map((model) => (
                    <button
                      key={model.id}
                      onClick={() => {
                        setSelectedModel(model.id);
                        setIsModelDropdownOpen(false);
                      }}
                      className={`w-full px-4 py-3 text-left hover:bg-slate-800/50 transition-colors ${
                        selectedModel === model.id ? 'bg-brand-500/10 border-l-4 border-brand-500' : ''
                      }`}
                      role="option"
                      aria-selected={selectedModel === model.id}
                    >
                      <div className="flex items-center gap-2">
                        <div className={`w-2 h-2 rounded-full ${
                          model.category === 'classification' ? 'bg-blue-400' :
                          model.category === 'detection' ? 'bg-green-400' :
                          model.category === 'segmentation' ? 'bg-purple-400' :
                          'bg-pink-400'
                        }`} />
                        <div className="flex-1">
                          <div className="font-medium flex items-center gap-2">
                            {model.name}
                            {model.recommended && (
                              <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-300 rounded">Recommended</span>
                            )}
                          </div>
                          <div className="text-sm text-slate-400">{model.description}</div>
                          <div className="text-xs text-slate-500 capitalize">{model.category}</div>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
          
          {/* Server Status */}
          <ServerStatus
            status={health.ok ? 'green' : 'yellow'}
            responseTime={latency}
            message={health.ok ? '' : 'Service may be cold starting'}
            onRetry={() => setHealth({ ok: true, message: '' })}
          />
        </div>
        {!health.ok && (
          <div className="flex items-center gap-2 p-3 rounded-xl border border-yellow-200/30 bg-yellow-500/10 text-yellow-100" role="status" aria-live="polite">
            <AlertTriangle size={18} />
            <span>{health.message || 'Backend may be cold starting. First request can take up to a minute on free tier.'}</span>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="flex flex-col sm:flex-row gap-1 p-1 justify-stretch rounded-xl bg-slate-900/50 border border-white/10">
          <button
            onClick={() => handleTabChange('image')}
            className={`flex grow items-center justify-center w-auto gap-2 px-3 sm:px-4 py-2 rounded-lg font-medium transition-all text-sm sm:text-base ${
              activeTab === 'image'
                ? 'bg-brand-500 text-white shadow-lg'
                : 'text-slate-400 bg-white/10 hover:text-white hover:bg-white/20'
            }`}
          >
            <ImageIcon className="w-4 h-4 sm:w-[18px] sm:h-[18px] flex-shrink-0" />
            <span className="hidden sm:inline">Image Processing</span>
            <span className="sm:hidden">Image</span>
          </button>
          <button
            onClick={() => handleTabChange('video')}
            className={`flex grow items-center justify-center w-auto gap-2 px-3 sm:px-4 py-2 rounded-lg font-medium transition-all text-sm sm:text-base ${
              activeTab === 'video'
                ? 'bg-brand-500 text-white shadow-lg'
                : 'text-slate-400 bg-white/10 hover:text-white hover:bg-white/20'
            }`}
          >
            <Video className="w-4 h-4 sm:w-[18px] sm:h-[18px] flex-shrink-0" />
            <span className="hidden sm:inline">Video Processing</span>
            <span className="sm:hidden">Video</span>
          </button>
          <button
            onClick={() => handleTabChange('stream')}
            className={`flex grow items-center justify-center w-auto gap-2 px-3 sm:px-4 py-2 rounded-lg font-medium transition-all text-sm sm:text-base ${
              activeTab === 'stream'
                ? 'bg-brand-500 text-white shadow-lg'
                : 'text-slate-400 bg-white/10 hover:text-white hover:bg-white/20'
            }`}
          >
            <Radio className="w-4 h-4 sm:w-[18px] sm:h-[18px] flex-shrink-0" />
            <span className="hidden sm:inline">Stream Processing</span>
            <span className="sm:hidden">Stream</span>
          </button>
        </div>

        {/* Tab Content */}
        {activeTab === 'image' && (
          <>
            <section className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-4 sm:p-5 shadow-xl">
              <ImageUpload
                onDrop={onDrop}
                onRejected={onRejected}
                isDragging={isDragging}
                setIsDragging={setIsDragging}
                file={file}
                previewUrl={previewUrl}
                onPredict={predict}
                canPredict={canPredict}
                onReset={reset}
                onCameraModeChange={handleCameraModeChange}
                onOpenCamera={() => openCamera('image')}
              />
            </section>

            <section className="grid gap-3">
              {isLoading && (
                <div className="grid gap-2 place-items-center text-center p-4">
                  <LoadingSpinner />
                  <p className="text-slate-400 text-sm sm:text-base">Processing... This may take a bit if the backend is waking up.</p>
                  {uploadProgress > 0 && (
                    <div className="h-2 w-full max-w-xs rounded-full bg-white/10 overflow-hidden" aria-label="Upload progress" aria-valuenow={uploadProgress} aria-valuemin={0} aria-valuemax={100}>
                      <div className="h-full bg-gradient-to-r from-brand-500 to-blue-300" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  )}
                </div>
              )}
              {!!error && (
                <div className="flex items-start gap-2 p-3 rounded-xl border border-red-200/30 bg-red-500/10 text-red-100" role="alert">
                  <AlertTriangle className="w-4 h-4 sm:w-[18px] sm:h-[18px] flex-shrink-0 mt-0.5" />
                  <span className="text-sm sm:text-base">{error}</span>
                </div>
              )}
              {!isLoading && !error && results && (
                <PredictionResults results={results} />
              )}
              {!isLoading && !error && results && results.predictions && results.predictions.length > 0 && previewUrl && (
                <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
                  <h3 className="text-lg font-semibold text-white mb-4">Image with Predictions</h3>
                  <div className="flex justify-center">
                    <div className="relative max-w-full">
                      <ImageOverlay
                        imageSrc={previewUrl}
                        predictions={results.predictions}
                        className="rounded-xl border border-white/10 bg-slate-950"
                        onError={(error) => {
                          console.error('ImageOverlay error:', error);
                          setError(error);
                        }}
                      />
                      <div className="absolute top-2 right-2">
                        <div className="px-2 py-1 bg-black/60 rounded-md text-xs text-white">
                          AI Processed
                        </div>
                      </div>
                    </div>
                  </div>
                  <p className="text-slate-400 text-sm mt-3 text-center">
                    Computer vision predictions overlaid on your original image
                  </p>
                </div>
              )}
            </section>
          </>
        )}

        {activeTab === 'video' && (
          <>
            <section className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
              <VideoUpload
                onDrop={onVideoDrop}
                onRejected={onRejected}
                isDragging={isDragging}
                setIsDragging={setIsDragging}
                file={videoFile}
                previewUrl={videoPreviewUrl}
                onProcess={processVideo}
                canProcess={canProcessVideo}
                onReset={resetVideo}
                onCameraModeChange={handleCameraModeChange}
                onOpenCamera={() => openCamera('video')}
              />
            </section>

            <section className="grid gap-3">
              {isVideoProcessing && (
                <div className="grid gap-2 place-items-center text-center p-4">
                  <LoadingSpinner label="Processing video..." />
                  <p className="text-slate-400">Analyzing video frames...</p>
                  {uploadProgress > 0 && (
                    <div className="h-2 w-full rounded-full bg-white/10 overflow-hidden" aria-label="Upload progress" aria-valuenow={uploadProgress} aria-valuemin={0} aria-valuemax={100}>
                      <div className="h-full bg-gradient-to-r from-brand-500 to-blue-300" style={{ width: `${uploadProgress}%` }} />
                    </div>
                  )}
                </div>
              )}
              {!!error && (
                <div className="flex items-center gap-2 p-3 rounded-xl border border-red-200/30 bg-red-500/10 text-red-100" role="alert">
                  <AlertTriangle size={18} />
                  <span>{error}</span>
                </div>
              )}
              {!isVideoProcessing && !error && videoFrames.length > 0 && (
                <VideoFrameResults 
                  frames={videoFrames} 
                  currentTime={currentVideoTime}
                  className="mb-4"
                />
              )}
              {!isVideoProcessing && !error && processedVideoUrl && (
                <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white">Processed Video with Predictions</h3>
                    <div className="flex gap-2">
                      <a
                        href={processedVideoUrl}
                        download="original-video.mp4"
                        className="px-3 py-1.5 rounded-md border border-white/20 hover:bg-white/10 text-sm text-slate-300"
                        title="Download original video"
                      >
                        Download Original
                      </a>
                      <VideoDownloader
                        videoSrc={processedVideoUrl}
                        frames={videoFrames}
                        fileName={videoFile?.name?.replace(/\.[^/.]+$/, "_with_boxes.webm") || "processed_video_with_boxes.webm"}
                        className="inline-block"
                      />
                    </div>
                  </div>
                  {!hideProcessedVideo ? (
                    <div className="flex justify-center">
                      <div className="relative max-w-full">
                        <VideoOverlay
                          videoSrc={processedVideoUrl}
                          frames={videoFrames}
                          className="rounded-xl border border-white/10 bg-slate-950"
                          onError={(error) => {
                            console.error('VideoOverlay error:', error);
                            setVideoPlaybackError(true);
                            setError(error);
                          }}
                          onTimeUpdate={(time) => setCurrentVideoTime(time)}
                        />
                        <div className="absolute top-2 right-2">
                          <div className="px-2 py-1 bg-black/60 rounded-md text-xs text-white">
                            AI Processed
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center p-8 border-2 border-dashed border-white/20 rounded-xl">
                      <p className="text-slate-400">
                        Video player hidden due to playback issues. Use the download button above to access the processed video.
                      </p>
                    </div>
                  )}
                  <div className="mt-3 text-center">
                    {videoPlaybackError && (
                      <div className="mb-2 p-2 rounded-md bg-yellow-500/10 border border-yellow-500/20">
                        <p className="text-yellow-400 text-sm">
                          ⚠️ Video playback failed in browser. The processing was successful - you can download the processed video above or view the prediction results below.
                        </p>
                      </div>
                    )}
                    <p className="text-slate-400 text-sm">
                      Computer vision predictions overlaid on video frames
                    </p>
                    {results?.total_frames && (
                      <p className="text-slate-500 text-xs mt-1">
                        Processed {results.processed_frames} of {results.total_frames} frames
                      </p>
                    )}
                  </div>
                </div>
              )}
            </section>
          </>
        )}

        {activeTab === 'stream' && (
          <section className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
            <StreamProcessor
              onStart={startStreamProcessing}
              onStop={stopStreamProcessing}
              isStreaming={isStreaming}
              onSettings={handleStreamSettings}
            />
          </section>
        )}
      </main>

      {/* Global Camera Modal */}
      <CameraModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onCapture={handleCameraCapture}
        mode={cameraMode}
        onModeChange={handleCameraModeChange}
      />

      <footer className="max-w-3xl mx-auto px-4 py-6 flex items-center justify-between text-slate-400">
        <div className="inline-flex gap-2 items-center">
          <Timer size={16} />
          <span>Backend: auto-detected • Uses health checks and retries</span>
        </div>
        <button className="text-blue-300 hover:underline bg-transparent border-0 cursor-pointer">Docs</button>
      </footer>
    </div>
  );
}
