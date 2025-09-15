// Utility function to detect mobile devices
export const isMobileDevice = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(
    navigator.userAgent
  ) || window.innerWidth <= 768;
};

// Utility function to check if device supports camera capture
export const supportsCameraCapture = (): boolean => {
  if (typeof window === 'undefined') return false;
  
  // Check if the device supports the capture attribute
  const input = document.createElement('input');
  input.type = 'file';
  return 'capture' in input;
};
