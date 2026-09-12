import { useState } from "react";

const formatTimestamp = (utcString) => {
    if (!utcString) return "-";
    try {
        const isoString = utcString.includes("T") ? utcString : utcString.replace(" ", "T") + "Z";
        const date = new Date(isoString);
        return isNaN(date.getTime()) ? utcString : date.toLocaleString();
    } catch {
        return utcString;
    }
};

export default function LiveHistory({ history = [], addToast }) {

    const [searchTerm, setSearchTerm] = useState("");
    const [filterPrediction, setFilterPrediction] = useState("All");
    const [currentPage, setCurrentPage] = useState(1);
    const [selectedFlow, setSelectedFlow] = useState(null);

    const pageSize = 10;

    // Filter predictions list for dropdown options
    const uniquePredictions = ["All", ...new Set(history.map(f => f.prediction).filter(Boolean))];

    // Apply Search and Filters
    const filteredHistory = history.filter(flow => {
        const matchesSearch = 
            (flow.src_ip || "").includes(searchTerm) ||
            (flow.dst_ip || "").includes(searchTerm) ||
            (flow.flow_id || "").includes(searchTerm) ||
            (flow.protocol != null && flow.protocol.toString().includes(searchTerm));

        const matchesPrediction = 
            filterPrediction === "All" || 
            flow.prediction === filterPrediction;

        return matchesSearch && matchesPrediction;
    });

    // Pagination calculations
    const totalItems = filteredHistory.length;
    const totalPages = Math.ceil(totalItems / pageSize) || 1;
    const startIndex = (currentPage - 1) * pageSize;
    const paginatedHistory = filteredHistory.slice(startIndex, startIndex + pageSize);

    const handleSearchChange = (e) => {
        setSearchTerm(e.target.value);
        setCurrentPage(1);
    };

    const handleFilterChange = (e) => {
        setFilterPrediction(e.target.value);
        setCurrentPage(1);
    };

    const triggerDownload = (format) => {
        window.open(`http://127.0.0.1:8000/live/download/${format}`, "_blank");
        if (addToast) {
            addToast(`Telemetry live history exported as ${format.toUpperCase()}.`, "info");
        }
    };

    const getProtocolName = (proto) => {
        if (proto === 6) return "TCP";
        if (proto === 17) return "UDP";
        if (proto === 1) return "ICMP";
        return proto != null ? `Other (${proto})` : "UNKNOWN";
    };

    return (

        <div className="live-history">

            <div className="history-controls">
                <h2>Real-time SNIFFER History</h2>

                <div className="history-controls-left">
                    <input
                        type="text"
                        placeholder="Search IP, Port or Protocol..."
                        value={searchTerm}
                        onChange={handleSearchChange}
                        className="search-input"
                    />

                    <select
                        value={filterPrediction}
                        onChange={handleFilterChange}
                        className="filter-select"
                    >
                        {uniquePredictions.map(pred => (
                            <option key={pred} value={pred}>
                                {pred === "All" ? "Filter: All Predictions" : `Prediction: ${pred}`}
                            </option>
                        ))}
                    </select>

                    <button 
                        onClick={() => triggerDownload("csv")} 
                        className="btn-export"
                        title="Download active database table as CSV"
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        Export CSV
                    </button>

                    <button 
                        onClick={() => triggerDownload("json")} 
                        className="btn-export"
                        title="Download active database table as JSON"
                    >
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                            <polyline points="7 10 12 15 17 10" />
                            <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        Export JSON
                    </button>
                </div>
            </div>

            <div className="history-table-wrapper">
                <table className="history-table">
                    <thead>
                        <tr>
                            <th>Time Stamp</th>
                            <th>Source Address</th>
                            <th>Destination Address</th>
                            <th>Protocol</th>
                            <th>Classification</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {paginatedHistory.length > 0 ? (
                            paginatedHistory.map((flow) => {
                                const isThreat = flow.prediction !== "BENIGN";
                                return (
                                    <tr 
                                        key={flow.id} 
                                        className={isThreat ? "row-alert" : ""}
                                        onClick={() => setSelectedFlow(flow)}
                                    >
                                        <td>{formatTimestamp(flow.created_at)}</td>
                                        <td>{flow.src_ip}:{flow.src_port}</td>
                                        <td>{flow.dst_ip}:{flow.dst_port}</td>
                                        <td>{getProtocolName(flow.protocol)}</td>
                                        <td>
                                            {isThreat && (
                                                <span className="threat-icon-pulse" title="Security anomaly alert!">⚠️</span>
                                            )}
                                            <span className={`badge-label ${isThreat ? "threat" : "benign"}`}>
                                                {flow.prediction || "UNKNOWN"}
                                            </span>
                                        </td>
                                        <td className={`confidence-val ${isThreat ? "alert" : "benign"}`}>
                                            {flow.confidence != null ? `${(flow.confidence * 100).toFixed(1)}%` : "-"}
                                        </td>
                                    </tr>
                                );
                            })
                        ) : (
                            <tr>
                                <td colSpan="6" style={{ textAlign: "center", color: "var(--text-muted)", padding: "40px" }}>
                                    No matching flow records detected in logs.
                                </td>
                            </tr>
                        )}
                    </tbody>
                </table>
            </div>

            {/* Pagination Controls */}
            {totalItems > 0 && (
                <div className="history-pagination">
                    <span className="pagination-info">
                        Showing {startIndex + 1} to {Math.min(startIndex + pageSize, totalItems)} of {totalItems} logs
                    </span>

                    <div className="pagination-btns">
                        <button
                            className="btn-pagination"
                            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                            disabled={currentPage === 1}
                        >
                            &lt; Prev
                        </button>
                        <span style={{ fontSize: "0.8rem", color: "var(--text-secondary)" }}>
                            Page {currentPage} of {totalPages}
                        </span>
                        <button
                            className="btn-pagination"
                            onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                            disabled={currentPage === totalPages}
                        >
                            Next &gt;
                        </button>
                    </div>
                </div>
            )}

            {/* Detailed Flow Telemetry Modal */}
            <div className={`modal-overlay ${selectedFlow ? "open" : ""}`} onClick={() => setSelectedFlow(null)}>
                {selectedFlow && (
                    <div className="modal-container" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h3>Flow Packet Telemetry Profile</h3>
                            <button className="btn-modal-close" onClick={() => setSelectedFlow(null)}>
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                    <line x1="18" y1="6" x2="6" y2="18" />
                                    <line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>
                        <div className="modal-body">
                            <div className="detail-grid">
                                <div className="detail-item" style={{ gridColumn: "span 2" }}>
                                    <span className="detail-label">Flow Identifier</span>
                                    <span className="detail-value">{selectedFlow.flow_id}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Source Node</span>
                                    <span className="detail-value">{selectedFlow.src_ip}:{selectedFlow.src_port}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Destination Node</span>
                                    <span className="detail-value">{selectedFlow.dst_ip}:{selectedFlow.dst_port}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Protocol Layer</span>
                                    <span className="detail-value">{getProtocolName(selectedFlow.protocol)}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Session Duration</span>
                                    <span className="detail-value">{selectedFlow.duration != null ? `${selectedFlow.duration.toFixed(6)}s` : "-"}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Total Volume</span>
                                    <span className="detail-value">{(selectedFlow.bytes || 0).toLocaleString()} Bytes</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Total Packets</span>
                                    <span className="detail-value">{(selectedFlow.packets || 0).toLocaleString()}</span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Classification</span>
                                    <span className={`badge-label detail-value badge ${selectedFlow.prediction === "BENIGN" ? "benign" : "threat"}`}>
                                        {selectedFlow.prediction || "UNKNOWN"}
                                    </span>
                                </div>
                                <div className="detail-item">
                                    <span className="detail-label">Classifier Confidence</span>
                                    <span className={`detail-value confidence-val ${selectedFlow.prediction === "BENIGN" ? "benign" : "alert"}`}>
                                        {selectedFlow.confidence != null ? `${(selectedFlow.confidence * 100).toFixed(2)}%` : "-"}
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button className="btn-modal-action" onClick={() => setSelectedFlow(null)}>
                                Close Inspection
                            </button>
                        </div>
                    </div>
                )}
            </div>

        </div>

    );

}