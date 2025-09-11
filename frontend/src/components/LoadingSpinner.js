import React from 'react';
import PropTypes from 'prop-types';

const LoadingSpinner = ({ label = 'Loading...', variant = 'default', etaSec, cancelable = false, onCancel }) => {
  return (
    <div style={{ display: 'grid', gap: 8, justifyItems: 'center' }}>
      <div className="spinner" role="status" aria-live="polite" aria-label={label}>
        <div className="dot"></div>
        <div className="dot"></div>
        <div className="dot"></div>
      </div>
      <div className="muted" style={{ fontSize: 14 }}>
        {label}{typeof etaSec === 'number' ? ` • ~${Math.ceil(etaSec)}s` : ''}
      </div>
      {cancelable && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onCancel}>Cancel</button>
      )}
    </div>
  );
};

LoadingSpinner.propTypes = {
  label: PropTypes.string,
  variant: PropTypes.oneOf(['default', 'uploading', 'processing', 'connecting']),
  etaSec: PropTypes.number,
  cancelable: PropTypes.bool,
  onCancel: PropTypes.func
};

export default LoadingSpinner;
