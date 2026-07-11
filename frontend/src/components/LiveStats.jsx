export default function LiveStats({ stats = {} }) {

    const totalFlows = stats.total_flows || 0;
    const activeFlows = stats.active_flows || 0;
    const packetsPerSec = stats.avg_packets_per_second || 0;
    const bytesPerSec = stats.avg_bytes_per_second || 0;
    const totalPackets = stats.total_packets || 0;
    const totalBytes = stats.total_bytes || 0;
    const benignCount = stats.BENIGN || 0;

    // Sum up any key that is not metadata to get the total attack count
    let attackCount = 0;
    Object.entries(stats).forEach(([key, val]) => {
        const k = key.toLowerCase().trim();
        if (![
            "total_flows",
            "total_packets",
            "total_bytes",
            "avg_packets_per_second",
            "avg_bytes_per_second",
            "active_flows",
            "benign",
            "protocols"
        ].includes(k) && typeof val === "number") {
            attackCount += val;
        }
    });

    const formatBytes = (bytes) => {
        if (bytes === 0) return "0 Bytes";
        const k = 1024;
        const sizes = ["Bytes", "KB", "MB", "GB", "TB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    };

    const formatBps = (bps) => {
        if (bps === 0) return "0 B/s";
        const k = 1024;
        const sizes = ["B/s", "KB/s", "MB/s", "GB/s"];
        const i = Math.floor(Math.log(bps) / Math.log(k));
        return parseFloat((bps / Math.pow(k, i)).toFixed(1)) + " " + sizes[i];
    };

    const getBenignPercentage = () => {
        if (totalFlows === 0) return "0.0%";
        return ((benignCount / totalFlows) * 100).toFixed(1) + "%";
    };

    const getAttackPercentage = () => {
        if (totalFlows === 0) return "0.0%";
        return ((attackCount / totalFlows) * 100).toFixed(1) + "%";
    };

    const cardsData = [
        {
            key: "total_flows",
            title: "Total Flows",
            value: totalFlows.toLocaleString(),
            percent: "Cumulative DB",
            className: "total",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
                    <line x1="4" y1="22" x2="4" y2="15" />
                </svg>
            )
        },
        {
            key: "active_flows",
            title: "Active Flows",
            value: activeFlows.toLocaleString(),
            percent: "Sniffing Memory",
            className: "active-flows",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
                    <path d="M22 12A10 10 0 0 0 12 2v10z" />
                </svg>
            )
        },
        {
            key: "pps",
            title: "Packets / sec",
            value: packetsPerSec.toLocaleString() + " pps",
            percent: "Average rate",
            className: "portscan",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
            )
        },
        {
            key: "bps",
            title: "Bytes / sec",
            value: formatBps(bytesPerSec),
            percent: "Bandwidth usage",
            className: "other-attacks",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <polyline points="19 12 12 19 5 12" />
                </svg>
            )
        },
        {
            key: "total_packets",
            title: "Total Packets",
            value: totalPackets.toLocaleString(),
            percent: "Received packets",
            className: "total",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
                    <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
                    <line x1="6" y1="6" x2="6.01" y2="6" />
                    <line x1="6" y1="18" x2="6.01" y2="18" />
                </svg>
            )
        },
        {
            key: "total_bytes",
            title: "Total Volume",
            value: formatBytes(totalBytes),
            percent: "Payload size",
            className: "total",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
                    <line x1="12" y1="22.08" x2="12" y2="12" />
                </svg>
            )
        },
        {
            key: "benign",
            title: "Benign Flows",
            value: benignCount.toLocaleString(),
            percent: `${getBenignPercentage()} of total`,
            className: "benign",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
            )
        },
        {
            key: "attacks",
            title: "Detected Threats",
            value: attackCount.toLocaleString(),
            percent: `${getAttackPercentage()} of total`,
            className: "ddos",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
            )
        }
    ];

    return (

        <div className="live-stats">

            <h2>Real-time Threat Intelligence Metrics</h2>

            <div className="live-stats-grid">
                {cardsData.map((card) => (
                    <div key={card.key} className={`stat-card ${card.className}`}>
                        <div className="stat-header">
                            <span>{card.title}</span>
                            <span className="stat-icon">{card.icon}</span>
                        </div>
                        <h1 className="stat-value">{card.value}</h1>
                        
                        <div className="stat-bar-container">
                            <div 
                                className="stat-bar" 
                                style={{ width: card.key === "benign" ? getBenignPercentage() : (card.key === "attacks" ? getAttackPercentage() : "100%") }}
                            ></div>
                        </div>
                        <span className="stat-percentage">{card.percent}</span>
                    </div>
                ))}
            </div>

        </div>

    );

}