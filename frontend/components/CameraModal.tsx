import React, { useRef, useState, useEffect } from 'react';
import { Camera, Video, X } from 'lucide-react';

type CameraModalProps = {
  isOpen: boolean;
  onClose: () => void;
  onCapture: (file: File) => void;
  mode?: 'image' | 'video'; // New prop to determine mode
  onModeChange?: (mode: 'image' | 'video') => void; // New callback for mode changes
};

const CameraModal: React.FC<CameraModalProps> = ({ isOpen, onClose, onCapture, mode = 'image', onModeChange }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string>('');
  const [currentMode, setCurrentMode] = useState<'image' | 'video'>(mode);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);

  // Start camera when modal opens
  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => stopCamera();
  }, [isOpen]);

  // Update mode when prop changes, but only if modal is opening
  useEffect(() => {
    if (isOpen) {
      setCurrentMode(mode);
    }
  }, [mode, isOpen]);

  // Recording timer
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording) {
      setRecordingDuration(0);
      interval = setInterval(() => {
        setRecordingDuration(prev => prev + 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
      if (!isRecording) setRecordingDuration(0);
    };
  }, [isRecording]);

  const startCamera = async () => {
    try {
      setError('');
      const mediaStream = await navigator.mediaDevices.getUserMedia({ 
        video: true, 
        audio: false // Disable audio recording
      });
      
      console.log('Camera stream obtained:', {
        videoTracks: mediaStream.getVideoTracks().length,
        audioTracks: mediaStream.getAudioTracks().length,
        tracks: mediaStream.getTracks().map(track => ({
          kind: track.kind,
          enabled: track.enabled,
          readyState: track.readyState
        }))
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        videoRef.current.play();
      }
      
      setStream(mediaStream);
    } catch (err) {
      setError('Camera access denied or not available');
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const takePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');

    if (!context) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current video frame to canvas
    context.drawImage(video, 0, 0);

    // Convert to file
    canvas.toBlob((blob) => {
      if (blob) {
        const file = new File([blob], `photo-${Date.now()}.jpg`, { type: 'image/jpeg' });
        onCapture(file);
        onClose();
      }
    }, 'image/jpeg', 0.9);
  };

  const startVideoRecording = () => {
    if (!stream) return;

    try {
      // Reset recorded chunks
      recordedChunksRef.current = [];
      
      // Use WebM since MP4 recording is rarely supported in browsers
      let options;
      if (MediaRecorder.isTypeSupported('video/webm;codecs=vp8')) {
        options = { mimeType: 'video/webm;codecs=vp8' };
      } else if (MediaRecorder.isTypeSupported('video/webm')) {
        options = { mimeType: 'video/webm' };
      } else if (MediaRecorder.isTypeSupported('video/mp4')) {
        options = { mimeType: 'video/mp4' };
      } else {
        options = {}; // Use default
      }

      console.log('Using MediaRecorder options:', options);
      const mediaRecorder = new MediaRecorder(stream, options);
      
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        console.log('Data available:', event.data.size, 'bytes');
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        console.log('Recording stopped, chunks collected:', recordedChunksRef.current.length);
        
        const recordedMimeType = options.mimeType || 'video/webm';
        
        // Create blob from all collected chunks
        const blob = new Blob(recordedChunksRef.current, { type: recordedMimeType });
        
        console.log('Created blob:', { 
          size: blob.size, 
          type: blob.type,
          chunks: recordedChunksRef.current.length
        });
        
        if (blob.size === 0) {
          console.error('No video data recorded!');
          setError('No video data was recorded. Try recording for at least 1 second.');
          setIsRecording(false);
          return;
        }
        
        if (blob.size < 1000) {
          console.warn('Very small video file recorded:', blob.size, 'bytes');
        }
        
        // Create file with correct extension and MIME type based on what was actually recorded
        let fileExtension = 'mp4';
        let fileMimeType = 'video/mp4';
        
        if (recordedMimeType.includes('webm')) {
          fileExtension = 'webm';
          fileMimeType = 'video/webm';
        } else if (recordedMimeType.includes('mp4')) {
          fileExtension = 'mp4';
          fileMimeType = 'video/mp4';
        }
        
        const file = new File([blob], `video-${Date.now()}.${fileExtension}`, { type: fileMimeType });
        
        console.log('Created video file:', { 
          name: file.name, 
          size: file.size, 
          type: file.type,
          recordedAs: recordedMimeType,
          finalExtension: fileExtension,
          finalMimeType: fileMimeType
        });
        
        // Test if the video blob can be played by creating a URL
        const testUrl = URL.createObjectURL(blob);
        console.log('Test video URL created:', testUrl);
        
        // Additional validation: try to create a video element to test playback
        const testVideo = document.createElement('video');
        testVideo.src = testUrl;
        testVideo.onloadedmetadata = () => {
          console.log('✅ Recorded video is valid:', {
            duration: testVideo.duration,
            videoWidth: testVideo.videoWidth,
            videoHeight: testVideo.videoHeight
          });
          URL.revokeObjectURL(testUrl); // Clean up test URL
        };
        testVideo.onerror = (e) => {
          console.error('❌ Recorded video validation failed:', e);
          URL.revokeObjectURL(testUrl); // Clean up test URL
        };
        
        onCapture(file);
        onClose();
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setError('Recording failed');
        setIsRecording(false);
      };

      mediaRecorder.onstart = () => {
        console.log('Recording started');
        setIsRecording(true);
      };

      // Start recording with a timeslice to ensure regular data availability
      mediaRecorder.start(1000); // Collect data every 1 second
      console.log('Started recording with state:', mediaRecorder.state);
    } catch (err) {
      setError('Video recording not supported');
      console.error('Recording error:', err);
    }
  };

  const stopVideoRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      console.log('Stopping recording, current state:', mediaRecorderRef.current.state);
      
      if (mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
      
      setIsRecording(false);
    }
  };

  const handleModeSwitch = (newMode: 'image' | 'video') => {
    setCurrentMode(newMode);
    // Notify parent component about mode change
    if (onModeChange) {
      onModeChange(newMode);
    }
    // Keep the existing stream running - no need to restart camera
    // The stream works for both photo capture and video recording
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 rounded-2xl border border-white/10 p-6 max-w-2xl w-full">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">
            {currentMode === 'image' ? 'Take a Photo' : 'Record a Video'}
          </h3>
          <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg">
            <X size={20} className="text-slate-400" />
          </button>
        </div>

        {/* Mode Switch */}
        <div className="flex gap-2 mb-4 p-1 bg-slate-800 rounded-lg">
          <button
            onClick={() => handleModeSwitch('image')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
              currentMode === 'image' 
                ? 'bg-brand-500 text-white' 
                : 'text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
          >
            <Camera size={16} />
            Photo
          </button>
          <button
            onClick={() => handleModeSwitch('video')}
            className={`flex-1 flex items-center justify-center gap-2 py-2 px-4 rounded-md transition-colors ${
              currentMode === 'video' 
                ? 'bg-blue-500 text-white' 
                : 'text-slate-400 hover:text-white hover:bg-slate-700'
            }`}
          >
            <Video size={16} />
            Video
          </button>
        </div>

        <canvas ref={canvasRef} className="hidden" />

        {error ? (
          <div className="text-center py-8">
            <p className="text-red-400 mb-4">{error}</p>
            <button onClick={startCamera} className="px-4 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-md">
              Try Again
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="bg-black rounded-xl overflow-hidden relative" style={{ aspectRatio: '16/9' }}>
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
              {/* Recording indicator */}
              {isRecording && (
                <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-500 text-white px-3 py-1 rounded-full">
                  <div className="w-2 h-2 bg-white rounded-full animate-pulse"></div>
                  REC
                </div>
              )}
            </div>

            <div className="flex gap-3 justify-center">
              {currentMode === 'image' ? (
                <button
                  onClick={takePhoto}
                  disabled={!stream}
                  className="px-6 py-3 bg-brand-500 hover:bg-brand-600 disabled:opacity-50 text-white rounded-lg font-medium flex items-center gap-2"
                >
                  <Camera size={20} />
                  Take Photo
                </button>
              ) : (
                <button
                  onClick={isRecording ? stopVideoRecording : startVideoRecording}
                  disabled={!stream}
                  className={`px-6 py-3 ${
                    isRecording 
                      ? 'bg-red-500 hover:bg-red-600' 
                      : 'bg-blue-500 hover:bg-blue-600'
                  } disabled:opacity-50 text-white rounded-lg font-medium flex items-center gap-2`}
                >
                  <Video size={20} />
                  {isRecording ? 'Stop Recording' : 'Start Recording'}
                </button>
              )}
              <button
                onClick={onClose}
                className="px-6 py-3 border border-white/20 hover:bg-white/10 text-white rounded-lg font-medium"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraModal;
