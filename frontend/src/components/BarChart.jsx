import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Tooltip,
    Legend
} from "chart.js";
import { Bar } from "react-chartjs-2";

ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Tooltip,
    Legend
);

function BarChart({ result }) {
    if (!result) return null;

    // Color code based on threat category
    const backgroundColors = Object.keys(result.summary).map(label => {
        const name = label.toLowerCase();
        if (name.includes("benign")) return "#00e676";
        if (name.includes("ddos")) return "#ff1744";
        if (name.includes("portscan") || name.includes("port scan") || name.includes("port_scan")) return "#ff9100";
        if (name.includes("bot")) return "#d500f9";
        return "#00e5ff"; // Default neon cyan
    });

    const hoverBackgroundColors = backgroundColors.map(color => color + "bb");

    const data = {
        labels: Object.keys(result.summary),
        datasets: [
            {
                label: "Threat Counts",
                data: Object.values(result.summary),
                backgroundColor: backgroundColors,
                hoverBackgroundColor: hoverBackgroundColors,
                borderWidth: 1,
                borderColor: "rgba(255, 255, 255, 0.05)",
                borderRadius: 4
            }
        ]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                display: false
            },
            tooltip: {
                backgroundColor: "rgba(11, 16, 23, 0.95)",
                titleColor: "var(--neon-cyan)",
                bodyColor: "#f1f5f9",
                borderColor: "rgba(0, 229, 255, 0.2)",
                borderWidth: 1,
                padding: 12,
                displayColors: false,
                bodyFont: {
                    family: "'Inter', sans-serif",
                    size: 13
                },
                titleFont: {
                    family: "'Orbitron', sans-serif",
                    weight: "bold",
                    size: 13
                }
            }
        },
        scales: {
            x: {
                grid: {
                    color: "rgba(255, 255, 255, 0.03)",
                    drawTicks: true
                },
                ticks: {
                    color: "#94a3b8",
                    font: {
                        family: "'Inter', sans-serif",
                        size: 11
                    }
                }
            },
            y: {
                grid: {
                    color: "rgba(255, 255, 255, 0.03)",
                    drawTicks: true
                },
                ticks: {
                    color: "#94a3b8",
                    font: {
                        family: "'Inter', sans-serif",
                        size: 11
                    }
                }
            }
        }
    };

    return (
        <div className="chart-card card">
            <h2>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--neon-cyan)" }}>
                    <line x1="18" y1="20" x2="18" y2="10" />
                    <line x1="12" y1="20" x2="12" y2="4" />
                    <line x1="6" y1="20" x2="6" y2="14" />
                </svg>
                Threat Category Counts
            </h2>
            <div className="chart-container-inner">
                <Bar data={data} options={options} />
            </div>
        </div>
    );
}

export default BarChart;