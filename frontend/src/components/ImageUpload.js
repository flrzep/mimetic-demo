import React, { useMemo, useRef } from 'react';
import PropTypes from 'prop-types';
import { useDropzone } from 'react-dropzone';
import { ImagePlus, Upload, Trash2 } from 'lucide-react';

const MAX_SIZE = 10 * 1024 * 1024; // 10MB

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

const ImageUpload = ({ onDrop, onRejected, isDragging, setIsDragging, file, previewUrl, onPredict, canPredict, onReset }) => {
  const inputRef = useRef(null);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragReject
  } = useDropzone({
    onDrop,
    onDropRejected: onRejected,
    accept: {
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg']
    },
    maxSize: MAX_SIZE,
    multiple: false
  });

  const handleBrowse = () => inputRef.current?.click();

  const borderClass = useMemo(() => {
    if (isDragReject) return 'dropzone border-danger';
    if (isDragActive || isDragging) return 'dropzone border-active';
    return 'dropzone';
  }, [isDragActive, isDragReject, isDragging]);

  return (
    <div className="upload-wrapper">
      <div {...getRootProps({ className: borderClass, onDragEnter: () => setIsDragging(true), onDragLeave: () => setIsDragging(false) })}>
        <input {...getInputProps({ 'aria-label': 'Image upload dropzone' })} />
        <div className="dropzone-inner">
          {!previewUrl ? (
            <>
              <div className="icon-wrap">
                <ImagePlus size={36} />
              </div>
              <p className="dz-title">Drag & drop image here</p>
              <p className="dz-subtitle">PNG, JPG up to 10MB</p>
              <button type="button" className="btn btn-primary" onClick={handleBrowse} aria-label="Browse files">
                <Upload size={16} />
                <span>Browse Files</span>
              </button>
            </>
          ) : (
            <div className="preview">
              <img src={previewUrl} alt={file?.name || 'Selected image preview'} className="preview-img" />
              <div className="preview-meta">
                <p className="file-name" title={file?.name}>{file?.name}</p>
                <p className="file-size">{formatBytes(file?.size || 0)}</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="actions">
        <button
          type="button"
          className="btn btn-primary"
          onClick={onPredict}
          disabled={!canPredict}
          aria-disabled={!canPredict}
        >
          <Upload size={16} />
          <span>Predict</span>
        </button>
        <button
          type="button"
          className="btn btn-ghost"
          onClick={onReset}
          disabled={!file}
        >
          <Trash2 size={16} />
          <span>Clear</span>
        </button>
      </div>

      {/* Hidden input for explicit browse trigger */}
      <input
        ref={inputRef}
        type="file"
        accept="image/png, image/jpeg"
        className="hidden-input"
        onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])}
      />
    </div>
  );
};

ImageUpload.propTypes = {
  onDrop: PropTypes.func.isRequired,
  onRejected: PropTypes.func.isRequired,
  isDragging: PropTypes.bool.isRequired,
  setIsDragging: PropTypes.func.isRequired,
  file: PropTypes.object,
  previewUrl: PropTypes.string,
  onPredict: PropTypes.func.isRequired,
  canPredict: PropTypes.bool.isRequired,
  onReset: PropTypes.func.isRequired
};

export default ImageUpload;
