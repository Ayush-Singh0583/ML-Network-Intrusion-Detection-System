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

    const data = {

        labels: Object.keys(result.summary),

        datasets: [

            {

                label: "Attack Count",

                data: Object.values(result.summary),

                backgroundColor: "#00bcd4"

            }

        ]

    };

    return (

        <div className="chart-card">

            <h2>Attack Count</h2>

            <Bar data={data} />

        </div>

    );

}

export default BarChart;