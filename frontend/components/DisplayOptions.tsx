import React from 'react';

interface DisplayOptionsProps {
  showBoxes: boolean;
  setShowBoxes: (v: boolean) => void;
  showKeypoints: boolean;
  setShowKeypoints: (v: boolean) => void;
  boxThreshold: number; // 0..1
  setBoxThreshold: (v: number) => void;
  keypointThreshold: number; // 0..1
  setKeypointThreshold: (v: number) => void;
  disabled?: boolean;
}

const Slider: React.FC<{
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
}> = ({ label, value, onChange, min = 0, max = 1, step = 0.01, disabled }) => (
  <div className="grid gap-1">
    <div className="flex justify-between text-xs text-slate-400">
      <span>{label}</span>
      <span>{(value * 100).toFixed(0)}%</span>
    </div>
    <input
      type="range"
      aria-label={label}
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      disabled={disabled}
      className="w-full accent-brand-500 disabled:opacity-50"
    />
  </div>
);

const Toggle: React.FC<{
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}> = ({ label, checked, onChange, disabled }) => (
  <label className="flex items-center justify-between gap-3 p-2 rounded-lg bg-white/5 border border-white/10">
    <span className="text-sm text-slate-200">{label}</span>
    <input
      type="checkbox"
      className="h-4 w-4"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      disabled={disabled}
    />
  </label>
);

const DisplayOptions: React.FC<DisplayOptionsProps> = ({
  showBoxes,
  setShowBoxes,
  showKeypoints,
  setShowKeypoints,
  boxThreshold,
  setBoxThreshold,
  keypointThreshold,
  setKeypointThreshold,
  disabled = false,
}) => {
  return (
    <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-4 sm:p-5 shadow-xl">
      <h3 className="text-base font-semibold mb-3">Display Options</h3>
      <div className="grid gap-3">
        <div className="grid gap-2 sm:grid-cols-2">
          <Toggle label="Show bounding boxes" checked={showBoxes} onChange={setShowBoxes} disabled={disabled} />
          <Toggle label="Show keypoints" checked={showKeypoints} onChange={setShowKeypoints} disabled={disabled} />
        </div>
        <div className="grid gap-3 sm:grid-cols-2">
          <Slider label="Box confidence threshold" value={boxThreshold} onChange={setBoxThreshold} disabled={!showBoxes || disabled} />
          <Slider label="Keypoint score threshold" value={keypointThreshold} onChange={setKeypointThreshold} disabled={!showKeypoints || disabled} />
        </div>
        <p className="text-xs text-slate-500">
          These settings affect visualization only. Model inference results are unchanged.
        </p>
      </div>
    </div>
  );
};

export default DisplayOptions;
