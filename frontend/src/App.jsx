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

import "./styles/App.css";

function App() {

    const [file, setFile] = useState(null);

    const [result, setResult] = useState(null);

    const [loading, setLoading] = useState(false);

    async function uploadFile() {

        if (!file) {

            alert("Please choose a CSV");

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

        }

        catch {

            alert("Prediction Failed");

        }

        setLoading(false);

    }

    return (

        <div className="container">

            <Header />

            <UploadCard

                setFile={setFile}

                uploadFile={uploadFile}

                loading={loading}

            />

            <SummaryCards

                result={result}

            />

            <div className="dashboard-grid">

                <PieChart

                    result={result}

                />

                <BarChart

                    result={result}

                />

            </div>

            <ModelInfo />

            <DownloadButton

                result={result}

            />

            <Footer />

        </div>

    );

}

export default App;