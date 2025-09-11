import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ImagePlus, Upload, Trash2, AlertTriangle, Timer } from 'lucide-react';
import ImageUpload from './components/ImageUpload';
import PredictionResults from './components/PredictionResults';
import LoadingSpinner from './components/LoadingSpinner';
import { api, checkHealth, predictImage, connectWebSocket } from './utils/api';
import { ENDPOINTS } from './config';
import ServerStatus from './components/ServerStatus';

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState('');
  const [health, setHealth] = useState({ ok: true, message: '' });
  const [uploadProgress, setUploadProgress] = useState(0);
  const [ws, setWs] = useState(null);
  const [latency, setLatency] = useState(undefined);

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
    return () => {
      isMounted = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onDrop = useCallback((acceptedFiles) => {
    if (!acceptedFiles?.length) return;
    setError('');
    setResults(null);
    setFile(acceptedFiles[0]);
  }, []);

  const onRejected = useCallback((fileRejections) => {
    if (!fileRejections?.length) return;
    const first = fileRejections[0];
    const reasons = first.errors?.map(e => e.message).join(', ');
    setError(reasons || 'File rejected');
  }, []);

  const reset = useCallback(() => {
    setFile(null);
    setPreviewUrl('');
    setResults(null);
    setError('');
  }, []);

  const canPredict = useMemo(() => !!file && !isLoading, [file, isLoading]);

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  const predict = useCallback(async () => {
    if (!file) {
      setError('Please select an image first.');
      return;
    }
    setIsLoading(true);
    setError('');
    setResults(null);
    setUploadProgress(0);
    try {
      const start = performance.now();

      const maxAttempts = 3;
      let lastErr = null;
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
          const res = await predictImage(file, (e) => {
            if (e.total) setUploadProgress(Math.round((e.loaded / e.total) * 100));
          });

          const elapsed = (performance.now() - start) / 1000;
          const data = res.data;
          if (!data?.success) {
            throw new Error(data?.message || 'Prediction failed');
          }

          setResults({
            predictions: data.predictions || [],
            processing_time: data.processing_time ?? elapsed
          });
          lastErr = null;
          break;
        } catch (e) {
          lastErr = e;
          const retriable =
            e.code === 'ECONNABORTED' ||
            e.message?.toLowerCase().includes('network') ||
            [502, 503, 504].includes(e.response?.status);
          if (attempt < maxAttempts && retriable) {
            // Exponential backoff with jitter
            const delay = 1000 * Math.pow(2, attempt - 1) + Math.random() * 500;
            await sleep(delay);
            continue;
          }
          throw e;
        }
      }

      if (lastErr) throw lastErr;
    } catch (e) {
      const msg =
        e.code === 'ECONNABORTED'
          ? 'Request timed out. The backend may be waking up (Render cold start). Please try again in a few seconds.'
          : e.response?.data?.message || e.message || 'An error occurred during prediction.';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [file]);

  return (
    <div className="app-root">
      <header className="app-header">
        <div className="container">
          <h1 className="title">Computer Vision Demo</h1>
          <p className="subtitle">Upload an image to get AI predictions</p>
          <div style={{ marginTop: 10 }}>
            <ServerStatus
              status={health.ok ? 'green' : 'yellow'}
              responseTime={latency}
              message={health.ok ? '' : 'Service may be cold starting'}
              onRetry={() => {
                setHealth({ ok: true, message: '' });
              }}
            />
          </div>
        </div>
      </header>

      <main className="container main-grid">
        {!health.ok && (
          <div className="banner banner-warn" role="status" aria-live="polite">
            <AlertTriangle size={18} />
            <span>{health.message || 'Backend may be cold starting. First request can take up to a minute on free tier.'}</span>
          </div>
        )}

        <section className="card">
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
          />
        </section>

        <section className="results-section">
          {isLoading && (
            <div className="loading-panel">
              <LoadingSpinner />
              <p className="muted">Processing... This may take a bit if the backend is waking up.</p>
              {uploadProgress > 0 && (
                <div className="progress" aria-label="Upload progress" aria-valuenow={uploadProgress} aria-valuemin="0" aria-valuemax="100">
                  <div className="progress-bar" style={{ width: `${uploadProgress}%` }} />
                </div>
              )}
            </div>
          )}
          {!!error && (
            <div className="banner banner-error" role="alert">
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}
          {!isLoading && !error && results && (
            <PredictionResults results={results} />
          )}
        </section>
      </main>

      <footer className="container footer">
        <div className="footer-left">
          <Timer size={16} />
          <span className="muted">Backend: auto-detected • Uses health checks and retries</span>
        </div>
        <a className="link" href="#" rel="noreferrer">Docs</a>
      </footer>
    </div>
  );
}

export default App;
