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

    const data = {

        labels: Object.keys(result.summary),

        datasets: [

            {

                label: "Attack Distribution",

                data: Object.values(result.summary),

                backgroundColor: [

                    "#00e676",
                    "#ff1744",
                    "#ff9800",
                    "#03a9f4",
                    "#9c27b0",
                    "#ffeb3b",
                    "#00bcd4",
                    "#795548"

                ]

            }

        ]

    };

    return (

        <div className="chart-card">

            <h2>Attack Distribution</h2>

            <Pie data={data} />

        </div>

    );

}

export default PieChart;