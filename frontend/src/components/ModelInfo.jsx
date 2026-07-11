function ModelInfo() {
    return (
        <div className="model-card card">
            <h2>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--neon-cyan)" }}>
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                Model Intelligence & Diagnostics
            </h2>

            <div className="model-grid">
                <div className="model-metric">
                    <span className="model-metric-label">Classifier Engine</span>
                    <span className="model-metric-value active">Random Forest</span>
                </div>
                
                <div className="model-metric">
                    <span className="model-metric-label">Validated Accuracy</span>
                    <span className="model-metric-value high-acc">99.83%</span>
                </div>
                
                <div className="model-metric">
                    <span className="model-metric-label">Extracted Features</span>
                    <span className="model-metric-value">78 Dimensions</span>
                </div>
                
                <div className="model-metric">
                    <span className="model-metric-label">Target Classifications</span>
                    <span className="model-metric-value">15 Categories</span>
                </div>
                
                <div className="model-metric">
                    <span className="model-metric-label">Baseline Dataset</span>
                    <span className="model-metric-value">CIC IDS 2017</span>
                </div>
            </div>
        </div>
    );
}

export default ModelInfo;