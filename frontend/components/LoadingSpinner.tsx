import React from 'react';

type Props = {
  label?: string;
  variant?: 'default' | 'uploading' | 'processing' | 'connecting';
  etaSec?: number;
  cancelable?: boolean;
  onCancel?: () => void;
};

const LoadingSpinner: React.FC<Props> = ({
  label = 'Loading...',
  variant = 'default',
  etaSec,
  cancelable = false,
  onCancel,
}) => {
  return (
    <div style={{ display: 'grid', gap: 8, justifyItems: 'center' }} data-variant={variant}>
      <div className="spinner" role="status" aria-live="polite" aria-label={label}>
        <div className="dot"></div>
        <div className="dot"></div>
        <div className="dot"></div>
      </div>
      <div className="muted" style={{ fontSize: 14 }}>
        {label}
        {typeof etaSec === 'number' ? ` • ~${Math.ceil(etaSec)}s` : ''}
      </div>
      {cancelable && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  );
};

export default LoadingSpinner;
