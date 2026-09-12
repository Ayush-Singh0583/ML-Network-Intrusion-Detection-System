function Header() {
    return (
        <div className="header">
            <div className="header-banner">
                {/* Glowing Cyber Shield SVG */}
                <svg 
                    className="header-logo-svg"
                    width="44" 
                    height="44" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                    style={{ color: "var(--neon-cyan)" }}
                >
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <path d="M12 8v4"/>
                    <path d="M12 16h.01"/>
                </svg>
                <h1>SOC <span>Network Intrusion Detection System</span></h1>
            </div>
            
            <div className="header-meta">
                Random Forest Model • CIC IDS 2017 • FastAPI
            </div>

            <div className="system-ticker">
                <div className="ticker-item">
                    <span className="ticker-dot"></span>
                    <span>SECURITY CORES: ONLINE</span>
                </div>
                <div className="ticker-item" style={{ marginLeft: "15px" }}>
                    <span>THREAT LEVEL: NOMINAL</span>
                </div>
            </div>
        </div>
    );
}

export default Header;