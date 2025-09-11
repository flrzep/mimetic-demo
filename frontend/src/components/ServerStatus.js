import React from 'react';
import PropTypes from 'prop-types';

const dotColor = (status) => {
  if (status === 'green') return '#22c55e';
  if (status === 'yellow') return '#f59e0b';
  return '#ef4444';
};

const ServerStatus = ({ status, responseTime, message, onRetry }) => {
  return (
    <div className="server-status" role="status" aria-live="polite">
      <span className="status-dot" style={{ backgroundColor: dotColor(status) }} aria-hidden="true" />
      <span className="status-text">
        {status.toUpperCase()} {typeof responseTime === 'number' ? `• ${Math.round(responseTime)}ms` : ''}
      </span>
      {message ? <span className="muted">— {message}</span> : null}
      {onRetry && (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry} aria-label="Retry connection">Retry</button>
      )}
    </div>
  );
};

ServerStatus.propTypes = {
  status: PropTypes.oneOf(['green', 'yellow', 'red']).isRequired,
  responseTime: PropTypes.number,
  message: PropTypes.string,
  onRetry: PropTypes.func
};

export default ServerStatus;
