function DownloadButton({ result }) {
    if (!result) return null;

    return (
        <div className="action-bar">
            <a 
                href="http://127.0.0.1:8000/download" 
                style={{ textDecoration: "none" }}
            >
                <button className="cyber-btn" style={{ borderColor: "var(--neon-green)", color: "var(--neon-green)", boxShadow: "0 0 10px rgba(0, 230, 118, 0.1)" }}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                    </svg>
                    Download Prediction Report (CSV)
                </button>
            </a>
        </div>
    );
}

export default DownloadButton;