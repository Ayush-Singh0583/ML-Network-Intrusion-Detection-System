import { useState, useRef, useEffect } from "react";

const SCAN_STAGES = [
    "Uploading traffic telemetry file...",
    "Validating network headers...",
    "Parsing raw packet records...",
    // No feature count here: the served model uses what its bundle lists
    // (see the Model panel), not the 78 columns of the raw CSV.
    "Extracting flow features...",
    "Running Random Forest classifier...",
    "Aggregating class prediction counts...",
    "Generating threat summary vectors..."
];

// Rotating telemetry scanning statuses. Only mounted while an upload is in
// flight, so every upload starts again at the first stage without an effect
// having to reset state (that reset was a synchronous setState in an effect).
function ScannerOverlay() {
    const [stage, setStage] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setStage(s => (s + 1) % SCAN_STAGES.length);
        }, 1500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="scanner-overlay">
            <div className="scan-line"></div>
            <div className="cyber-loader"></div>
            <div className="loader-status-text">{SCAN_STAGES[stage]}</div>
        </div>
    );
}

function UploadCard({ file, setFile, uploadFile, loading }) {
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef(null);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const droppedFile = e.dataTransfer.files[0];
            // Case-insensitive, like the file picker's accept=".csv":
            // "FLOWS.CSV" used to be rejected on drop but accepted via browse.
            if (droppedFile.name.toLowerCase().endsWith(".csv")) {
                setFile(droppedFile);
            } else {
                alert("Only network CSV files are accepted!");
            }
        }
    };

    const handleChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const onButtonClick = (e) => {
        e.preventDefault();
        fileInputRef.current.click();
    };

    const formatBytes = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className="card" style={{ position: "relative" }}>
            {/* Loading Scanner Overlay */}
            {loading && <ScannerOverlay />}

            <h2>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--neon-cyan)" }}>
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="17 8 12 3 7 8" />
                    <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                Traffic Telemetry Ingestion Portal
            </h2>

            <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleChange}
                style={{ display: "none" }}
            />

            <div 
                className={`upload-dropzone ${dragActive ? "drag-active" : ""}`}
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={onButtonClick}
            >
                <svg 
                    className="upload-icon-svg"
                    width="48" 
                    height="48" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="1.5" 
                    strokeLinecap="round" 
                    strokeLinejoin="round"
                >
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                    <line x1="12" y1="18" x2="12" y2="12" />
                    <line x1="9" y1="15" x2="15" y2="15" />
                </svg>

                <p>Drag and drop network traffic <strong>CSV</strong> file here, or click to browse</p>
                <button type="button" className="file-select-btn">Select Log File</button>
            </div>

            {/* Selected File Details */}
            {file && (
                <div style={{ display: "flex", justifyContent: "center" }}>
                    <div className="file-info-bar">
                        <div className="file-info-details">
                            <svg 
                                className="file-info-icon"
                                width="18" 
                                height="18" 
                                viewBox="0 0 24 24" 
                                fill="none" 
                                stroke="currentColor" 
                                strokeWidth="2" 
                                strokeLinecap="round" 
                                strokeLinejoin="round"
                            >
                                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                            </svg>
                            <span title={file.name} style={{ fontWeight: 600 }}>{file.name}</span>
                            <span style={{ color: "var(--text-muted)", fontSize: "0.8rem" }}>({formatBytes(file.size)})</span>
                        </div>
                        <button 
                            className="file-remove-btn" 
                            onClick={(e) => {
                                e.stopPropagation();
                                setFile(null);
                            }}
                            title="Remove file"
                        >
                            &times;
                        </button>
                    </div>
                </div>
            )}

            <div style={{ textAlign: "center", marginTop: "25px" }}>
                <button
                    className="cyber-btn"
                    onClick={(e) => {
                        e.stopPropagation();
                        uploadFile();
                    }}
                    disabled={!file || loading}
                >
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    Analyze Telemetry Packets
                </button>
            </div>
        </div>
    );
}

export default UploadCard;