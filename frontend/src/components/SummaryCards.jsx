function SummaryCards({ result }) {
    if (!result) return null;

    const totalRecords = result.total_records || 0;
    let benign = 0;
    let ddos = 0;
    let portScan = 0;
    let bot = 0;
    let other = 0;

    // Categorize attack counts from backend response summary
    Object.entries(result.summary || {}).forEach(([attack, count]) => {
        const name = attack.toLowerCase().trim();
        if (name.includes("benign")) {
            benign += count;
        } else if (name.includes("ddos")) {
            ddos += count;
        } else if (name.includes("portscan") || name.includes("port scan") || name.includes("port_scan")) {
            portScan += count;
        } else if (name.includes("bot")) {
            bot += count;
        } else {
            other += count;
        }
    });

    const getPercentage = (count) => {
        if (totalRecords === 0) return "0.0%";
        return ((count / totalRecords) * 100).toFixed(1) + "%";
    };

    const cardsData = [
        {
            key: "total",
            title: "Total Records",
            value: totalRecords,
            percent: "100%",
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
            key: "benign",
            title: "Benign Traffic",
            value: benign,
            percent: getPercentage(benign),
            className: "benign",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                </svg>
            )
        },
        {
            key: "ddos",
            title: "DDoS Threats",
            value: ddos,
            percent: getPercentage(ddos),
            className: "ddos",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                    <line x1="12" y1="9" x2="12" y2="13" />
                    <line x1="12" y1="17" x2="12.01" y2="17" />
                </svg>
            )
        },
        {
            key: "portscan",
            title: "PortScan Activity",
            value: portScan,
            percent: getPercentage(portScan),
            className: "portscan",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10" />
                    <line x1="22" y1="12" x2="18" y2="12" />
                    <line x1="6" y1="12" x2="2" y2="12" />
                    <line x1="12" y1="6" x2="12" y2="2" />
                    <line x1="12" y1="22" x2="12" y2="18" />
                </svg>
            )
        },
        {
            key: "bot",
            title: "Botnet Vectors",
            value: bot,
            percent: getPercentage(bot),
            className: "bot",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
                    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
            )
        },
        {
            key: "other",
            title: "Other Attacks",
            value: other,
            percent: getPercentage(other),
            className: "other-attacks",
            icon: (
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z" />
                    <path d="M12 8v4" />
                    <path d="M12 16h.01" />
                </svg>
            )
        }
    ];

    return (
        <div className="stats-container">
            <h2>Real-time Traffic Classification Summary</h2>
            <div className="cards">
                {cardsData.map((card) => (
                    <div key={card.key} className={`stat-card ${card.className}`}>
                        <div className="stat-header">
                            <span>{card.title}</span>
                            <span className="stat-icon">{card.icon}</span>
                        </div>
                        <h1 className="stat-value">{card.value.toLocaleString()}</h1>
                        
                        <div className="stat-bar-container">
                            <div 
                                className="stat-bar" 
                                style={{ width: card.percent }}
                            ></div>
                        </div>
                        <span className="stat-percentage">{card.percent} of traffic</span>
                    </div>
                ))}
            </div>
        </div>
    );
}

export default SummaryCards;