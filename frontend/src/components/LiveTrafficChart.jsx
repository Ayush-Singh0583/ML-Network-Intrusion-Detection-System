import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Tooltip,
    Legend
} from "chart.js";

import { Line, Doughnut, Bar } from "react-chartjs-2";
import { useEffect, useState } from "react";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    BarElement,
    ArcElement,
    Tooltip,
    Legend
);

export default function LiveTrafficChart({ stats = {} }) {

    const MAX_POINTS = 30;

    const [labels, setLabels] = useState([]);
    const [ppsData, setPpsData] = useState([]);
    const [bpsData, setBpsData] = useState([]);

    useEffect(() => {

        const now = new Date().toLocaleTimeString();

        setLabels(old => {
            const updated = [...old, now];
            return updated.slice(-MAX_POINTS);
        });

        setPpsData(old => {
            const updated = [
                ...old,
                stats.avg_packets_per_second || 0
            ];
            return updated.slice(-MAX_POINTS);
        });

        setBpsData(old => {
            const updated = [
                ...old,
                stats.avg_bytes_per_second || 0
            ];
            return updated.slice(-MAX_POINTS);
        });

    }, [stats]);

    // Line Chart: Live Traffic
    const trafficData = {
        labels,
        datasets: [
            {
                label: "Packets/sec",
                data: ppsData,
                borderColor: "#00e676",
                backgroundColor: "rgba(0, 230, 118, 0.1)",
                fill: true,
                tension: 0.35
            },
            {
                label: "Bytes/sec",
                data: bpsData,
                borderColor: "#00e5ff",
                backgroundColor: "rgba(0, 229, 255, 0.1)",
                fill: true,
                tension: 0.35
            }
        ]
    };

    const trafficOptions = {
        responsive: true,
        animation: false,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: { color: "#94a3b8" }
            }
        },
        scales: {
            x: {
                grid: { color: "rgba(255, 255, 255, 0.02)" },
                ticks: { color: "#64748b" }
            },
            y: {
                grid: { color: "rgba(255, 255, 255, 0.02)" },
                ticks: { color: "#64748b" }
            }
        }
    };

    // Doughnut Chart: Threats
    const threatLabels = [];
    const threatValues = [];
    const threatColors = [];

    if (stats.BENIGN !== undefined && stats.BENIGN > 0) {
        threatLabels.push("BENIGN");
        threatValues.push(stats.BENIGN);
        threatColors.push("#00e676");
    }

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
        ].includes(k) && typeof val === "number" && val > 0) {
            threatLabels.push(key);
            threatValues.push(val);
            if (k.includes("ddos")) threatColors.push("#ff1744");
            else if (k.includes("portscan") || k.includes("port scan") || k.includes("port_scan")) threatColors.push("#ff9100");
            else if (k.includes("bot")) threatColors.push("#d500f9");
            else threatColors.push("#ffea00");
        }
    });

    const threatData = {
        labels: threatLabels.length > 0 ? threatLabels : ["No Data"],
        datasets: [
            {
                data: threatValues.length > 0 ? threatValues : [1],
                backgroundColor: threatColors.length > 0 ? threatColors : ["rgba(100, 116, 139, 0.2)"],
                borderColor: "#0b1017",
                borderWidth: 2
            }
        ]
    };

    const doughnutOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "right",
                labels: {
                    color: "#94a3b8",
                    font: { size: 10 },
                    boxWidth: 10
                }
            }
        }
    };

    // Bar Chart: Protocols
    const protocolCounts = stats.protocols || {};
    const protocolLabels = Object.keys(protocolCounts);
    const protocolValues = Object.values(protocolCounts);

    const protocolData = {
        labels: protocolLabels.length > 0 ? protocolLabels : ["TCP", "UDP", "ICMP", "Other"],
        datasets: [
            {
                label: "Flow Count",
                data: protocolValues.length > 0 ? protocolValues : [0, 0, 0, 0],
                backgroundColor: [
                    "rgba(0, 229, 255, 0.2)",
                    "rgba(213, 0, 249, 0.2)",
                    "rgba(255, 145, 0, 0.2)",
                    "rgba(100, 116, 139, 0.2)"
                ],
                borderColor: [
                    "#00e5ff",
                    "#d500f9",
                    "#ff9100",
                    "#64748b"
                ],
                borderWidth: 1.5
            }
        ]
    };

    const barOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            x: {
                grid: { color: "rgba(255, 255, 255, 0.01)" },
                ticks: { color: "#64748b", font: { size: 10 } }
            },
            y: {
                grid: { color: "rgba(255, 255, 255, 0.01)" },
                ticks: { color: "#64748b", font: { size: 10 } }
            }
        }
    };

    return (

        <div className="live-charts-grid">

            <div className="live-chart-container" style={{ gridColumn: "span 2" }}>
                <h2>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "#00e5ff" }}>
                        <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                    </svg>
                    Live Link Utilization Bandwidth
                </h2>
                <div style={{ height: "250px" }}>
                    <Line data={trafficData} options={trafficOptions} />
                </div>
            </div>

            <div className="live-chart-container">
                <h2>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "#ff1744" }}>
                        <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
                        <path d="M22 12A10 10 0 0 0 12 2v10z" />
                    </svg>
                    Threat Vectors Proportion
                </h2>
                <div style={{ height: "220px" }}>
                    <Doughnut data={threatData} options={doughnutOptions} />
                </div>
            </div>

            <div className="live-chart-container">
                <h2>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "#d500f9" }}>
                        <rect x="2" y="2" width="20" height="20" rx="2" ry="2" />
                        <line x1="3" y1="12" x2="21" y2="12" />
                    </svg>
                    Protocol Classification
                </h2>
                <div style={{ height: "220px" }}>
                    <Bar data={protocolData} options={barOptions} />
                </div>
            </div>

        </div>

    );

}