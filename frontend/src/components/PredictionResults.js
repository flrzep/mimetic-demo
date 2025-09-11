import React from 'react';
import PropTypes from 'prop-types';

function barWidth(p) {
  const pct = Math.max(0, Math.min(100, Math.round(p * 100)));
  return pct + '%';
}

const PredictionResults = ({ results }) => {
  const { predictions = [], processing_time } = results || {};

  return (
    <div className="card results-card">
      <div className="results-header">
        <h2>Results</h2>
        {typeof processing_time === 'number' && (
          <span className="muted">Processed in {processing_time.toFixed(2)}s</span>
        )}
      </div>
      {predictions.length === 0 ? (
        <p className="muted">No predictions available.</p>
      ) : (
        <ul className="predictions-list" role="list">
          {predictions.map((p, idx) => (
            <li key={idx} className="prediction-item">
              <div className="prediction-row">
                <span className="pred-class">Class {p.class_id}</span>
                <span className="pred-score">{(p.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="progress">
                <div className="progress-bar" style={{ width: barWidth(p.confidence) }} />
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

PredictionResults.propTypes = {
  results: PropTypes.shape({
    predictions: PropTypes.arrayOf(
      PropTypes.shape({
        class_id: PropTypes.number.isRequired,
        confidence: PropTypes.number.isRequired
      })
    ),
    processing_time: PropTypes.number
  })
};

export default PredictionResults;
