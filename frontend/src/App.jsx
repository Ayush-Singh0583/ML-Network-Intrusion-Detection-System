import { useState } from "react";
import axios from "axios";

import Header from "./components/Header";
import UploadCard from "./components/UploadCard";
import SummaryCards from "./components/SummaryCards";
import DownloadButton from "./components/DownloadButton";
import PieChart from "./components/PieChart";
import BarChart from "./components/BarChart";
import ModelInfo from "./components/ModelInfo";
import Footer from "./components/Footer";

import LiveDashboard from "./pages/LiveDashboard";

import "./styles/App.css";

function App() {

    const [page, setPage] = useState("csv");

    const [file, setFile] = useState(null);
    const [result, setResult] = useState(null);

    const [loading, setLoading] = useState(false);

    const [showToast, setShowToast] = useState(false);

    async function uploadFile() {

        if (!file) {

            alert("Please choose a CSV file first.");

            return;

        }

        const formData = new FormData();

        formData.append("file", file);

        setLoading(true);

        try {

            const response = await axios.post(

                "http://127.0.0.1:8000/predict_csv",

                formData

            );

            setResult(response.data);

            setShowToast(true);

            setTimeout(() => {

                setShowToast(false);

            }, 4000);

        }

        catch {

            alert("Analysis failed.");

        }

        setLoading(false);

    }

    return (

        <div className="container">

            <Header />

            <div className="soc-navigation-bar">
                <button
                    className={`soc-nav-btn ${page === "csv" ? "active" : ""}`}
                    onClick={() => setPage("csv")}
                >
                    <svg className="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                        <polyline points="14 2 14 8 20 8" />
                        <line x1="16" y1="13" x2="8" y2="13" />
                        <line x1="16" y1="17" x2="8" y2="17" />
                        <polyline points="10 9 9 9 8 9" />
                    </svg>
                    CSV File Telemetry
                </button>
                <button
                    className={`soc-nav-btn ${page === "live" ? "active" : ""}`}
                    onClick={() => setPage("live")}
                >
                    <svg className="nav-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
                    </svg>
                    Live SOC Monitor
                </button>
            </div>

            {

                page === "csv"

                &&

                <>

                    <UploadCard

                        file={file}

                        setFile={setFile}

                        uploadFile={uploadFile}

                        loading={loading}

                    />

                    <SummaryCards

                        result={result}

                    />

                    {

                        result &&

                        <div className="dashboard-grid">

                            <PieChart

                                result={result}

                            />

                            <BarChart

                                result={result}

                            />

                        </div>

                    }

                    <ModelInfo />

                    <DownloadButton

                        result={result}

                    />

                </>

            }

            {

                page === "live"

                &&

                <LiveDashboard />

            }

            <Footer />

            <div className="toast-container">

                <div className={`toast ${showToast ? "show" : ""}`}>

                    <svg

                        className="toast-icon"

                        width="20"

                        height="20"

                        viewBox="0 0 24 24"

                        fill="none"

                        stroke="currentColor"

                        strokeWidth="2"

                        strokeLinecap="round"

                        strokeLinejoin="round"

                    >

                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />

                        <polyline points="22 4 12 14.01 9 11.01" />

                    </svg>

                    <span className="toast-message">

                        Prediction complete!

                    </span>

                    <div className="toast-progress"></div>

                </div>

            </div>

        </div>

    );

}

export default App;