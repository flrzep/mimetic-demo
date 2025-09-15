import React from 'react';

function barWidth(p: number) {
  const pct = Math.max(0, Math.min(100, Math.round(p * 100)));
  return pct + '%';
}

export type Prediction = { class_id: number; confidence: number };
export type Results = { predictions: Prediction[]; processing_time?: number };

const PredictionResults: React.FC<{ results: Results }> = ({ results }) => {
  const { predictions = [], processing_time } = results || {} as Results;

  return (
    <div className="rounded-2xl border border-white/10 bg-gradient-to-b from-slate-900 to-slate-950 p-5 shadow-xl">
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-xl font-semibold">Results</h2>
        {typeof processing_time === 'number' && <span className="text-slate-400">Processed in {processing_time.toFixed(2)}s</span>}
      </div>
      {predictions.length === 0 ? (
        <p className="text-slate-400 mt-2">No predictions available.</p>
      ) : (
        <ul className="grid gap-3 mt-3">
          {predictions.map((p, idx) => (
            <li key={idx} className="grid gap-2">
              <div className="flex justify-between">
                <span className="font-semibold">Class {p.class_id}</span>
                <span className="text-slate-400">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="h-2 rounded-full bg-white/10 overflow-hidden border border-white/10">
                <div className="h-full bg-gradient-to-r from-brand-500 to-blue-300" style={{ width: barWidth(p.confidence) }} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default PredictionResults;
