import React from 'react';

type Status = 'green' | 'yellow' | 'red';

type Props = {
  status: Status;
  responseTime?: number;
  message?: string;
  onRetry?: () => void;
};

const ServerStatus: React.FC<Props> = ({ status, responseTime, message, onRetry }) => {
  const getStatusColors = (status: Status) => {
    if (status === 'green') return {
      bg: 'bg-green-100 dark:bg-green-900',
      text: 'text-green-800 dark:text-green-300',
      dot: 'bg-green-500'
    };
    if (status === 'yellow') return {
      bg: 'bg-yellow-100 dark:bg-yellow-900',
      text: 'text-yellow-800 dark:text-yellow-300',
      dot: 'bg-yellow-500'
    };
    return {
      bg: 'bg-red-100 dark:bg-red-900',
      text: 'text-red-800 dark:text-red-300',
      dot: 'bg-red-500'
    };
  };

  const colors = getStatusColors(status);
  const statusText = status === 'green' ? 'Available' : status === 'yellow' ? 'Warning' : 'Offline';

  return (
    <div className="server-status flex flex-col items-center md:items-end gap-2" role="status" aria-live="polite">
      <span className={`inline-flex items-center ${colors.bg} ${colors.text} text-xs font-medium px-2.5 py-0.5 rounded-full whitespace-nowrap`}>
        <span className={`w-2 h-2 me-1 ${colors.dot} rounded-full`}></span>
        {statusText} {typeof responseTime === 'number' ? `• ${Math.round(responseTime)}ms` : ''}
      </span>
      {message ? <span className="text-slate-400 text-xs sm:text-sm truncate">— {message}</span> : null}
      {onRetry && (
        <button 
          type="button" 
          className="px-3 py-1 text-xs font-medium text-slate-400 hover:text-white bg-transparent border border-slate-600 hover:border-slate-500 rounded-md transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-slate-500 focus:ring-offset-2 focus:ring-offset-slate-900 whitespace-nowrap" 
          onClick={onRetry} 
          aria-label="Retry connection"
        >
          Retry
        </button>
      )}
    </div>
  );
};

export default ServerStatus;
