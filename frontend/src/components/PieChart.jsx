import {
    Chart as ChartJS,
    ArcElement,
    Tooltip,
    Legend
} from "chart.js";
import { Pie } from "react-chartjs-2";

ChartJS.register(
    ArcElement,
    Tooltip,
    Legend
);

function PieChart({ result }) {
    if (!result) return null;

    const fallbackColors = [
        "#00e5ff", // Neon Cyan
        "#ffea00", // Neon Amber
        "#00b0ff", // Electric Blue
        "#ff3d00", // Bright Orange-Red
        "#ffd600"  // Gold
    ];
    let fallbackIdx = 0;

    // Coordinated cyber colors
    const backgroundColors = Object.keys(result.summary).map(label => {
        const name = label.toLowerCase();
        if (name.includes("benign")) return "#00e676"; // Green
        if (name.includes("ddos")) return "#ff1744"; // Red
        if (name.includes("portscan") || name.includes("port scan") || name.includes("port_scan")) return "#ff9100"; // Orange
        if (name.includes("bot")) return "#d500f9"; // Purple
        
        const col = fallbackColors[fallbackIdx % fallbackColors.length];
        fallbackIdx++;
        return col;
    });

    const hoverBackgroundColors = backgroundColors.map(color => color + "dd");

    const data = {
        labels: Object.keys(result.summary),
        datasets: [
            {
                label: "Distribution Ratio",
                data: Object.values(result.summary),
                backgroundColor: backgroundColors,
                hoverBackgroundColor: hoverBackgroundColors,
                borderWidth: 2,
                borderColor: "#0b1017"
            }
        ]
    };

    const options = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: "right",
                labels: {
                    color: "#94a3b8",
                    font: {
                        family: "'Inter', sans-serif",
                        size: 11
                    },
                    padding: 12,
                    usePointStyle: true,
                    pointStyle: "circle"
                }
            },
            tooltip: {
                backgroundColor: "rgba(11, 16, 23, 0.95)",
                titleColor: "var(--neon-cyan)",
                bodyColor: "#f1f5f9",
                borderColor: "rgba(0, 229, 255, 0.2)",
                borderWidth: 1,
                padding: 12,
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
        }
    };

    return (
        <div className="chart-card card">
            <h2>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--neon-cyan)" }}>
                    <path d="M21.21 15.89A10 10 0 1 1 8 2.83" />
                    <path d="M22 12A10 10 0 0 0 12 2v10z" />
                </svg>
                Threat Ratio Distribution
            </h2>
            <div className="chart-container-inner">
                <Pie data={data} options={options} />
            </div>
        </div>
    );
}

export default PieChart;