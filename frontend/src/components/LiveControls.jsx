import { useEffect, useState } from "react";
import axios from "axios";

const API = "http://127.0.0.1:8000";

export default function LiveControls({ addToast }) {

    const [running, setRunning] = useState(false);

    useEffect(() => {

        async function loadStatus() {

            try {

                const response = await axios.get(
                    `${API}/live/status`
                );

                setRunning(response.data.running);

            }

            catch (err) {

                console.error(err);

            }

        }

        loadStatus();

    }, []);

    async function startCapture() {

        try {

            const response = await axios.post(
                `${API}/live/start`
            );

            if (addToast) {
                addToast("Live network sniffing initialized.", "success");
            } else {
                alert(response.data.message);
            }

            setRunning(true);

        }

        catch (err) {

            console.error(err);

            if (addToast) {
                addToast("Unable to establish interface link.", "error");
            } else {
                alert("Unable to start live capture.");
            }

        }

    }

    async function stopCapture() {

        try {

            const response = await axios.post(
                `${API}/live/stop`
            );

            if (addToast) {
                addToast("Live sniffing session suspended.", "warning");
            } else {
                alert(response.data.message);
            }

            setRunning(false);

        }

        catch (err) {

            console.error(err);

            if (addToast) {
                addToast("Failed to safely teardown interface.", "error");
            } else {
                alert("Unable to stop live capture.");
            }

        }

    }

    return (

        <div className={`live-controls ${running ? "running" : ""}`}>

            <div className="control-header-wrapper">
                <h2>Live Traffic Telemetry Capture</h2>
                
                <div className={`status-badge ${running ? "active" : "stopped"}`}>
                    <span className={running ? "pulse-dot" : "stopped-dot"}></span>
                    <span>{running ? "SCANNING ACTIVE" : "SESSION SUSPENDED"}</span>
                </div>
            </div>

            <div className="btn-grid">

                <button
                    className="btn-soc start"
                    onClick={startCapture}
                    disabled={running}
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    Start Capture
                </button>

                <button
                    className="btn-soc stop"
                    onClick={stopCapture}
                    disabled={!running}
                >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                    </svg>
                    Stop Capture
                </button>

            </div>

        </div>

    );

}