import { useEffect, useState } from "react";
import axios from "axios";

import LiveStats from "../components/LiveStats";
import LiveHistory from "../components/LiveHistory";
import LiveControls from "../components/LiveControls";
import LiveTrafficChart from "../components/LiveTrafficChart";

const API = "http://127.0.0.1:8000";

export default function LiveDashboard() {

    const [stats, setStats] = useState({});
    const [history, setHistory] = useState([]);
    const [toasts, setToasts] = useState([]);
    const [lastFlowId, setLastFlowId] = useState(null);

    // Toast alert triggers
    const addToast = (message, type = "info") => {
        const id = Date.now();
        setToasts(old => [...old, { id, message, type }]);
        setTimeout(() => {
            setToasts(old => old.filter(t => t.id !== id));
        }, 4000);
    };

    async function loadData() {

        try {

            const statsResponse =
                await axios.get(`${API}/live/stats`);

            const historyResponse =
                await axios.get(`${API}/live/history`);

            setStats(statsResponse.data);
            setHistory(historyResponse.data);

            // Reactive Alert check: search for new flows in recent history with prediction != "BENIGN"
            if (historyResponse.data.length > 0) {
                const latestFlow = historyResponse.data[0];
                if (lastFlowId != null && latestFlow.id > lastFlowId) {
                    const newAttacks = historyResponse.data.filter(
                        flow => flow.id > lastFlowId && flow.prediction !== "BENIGN"
                    );
                    if (newAttacks.length > 0) {
                        addToast(`SECURITY ALERT: ${newAttacks.length} anomaly vectors detected!`, "error");
                    }
                }
                setLastFlowId(latestFlow.id);
            } else {
                setLastFlowId(null);
            }

        }

        catch (err) {

            console.error(err);

        }

    }

    useEffect(() => {

        loadData();

        const timer = setInterval(loadData, 2000);

        return () => clearInterval(timer);

    }, [lastFlowId]);

    return (

        <div className="live-page">

            <LiveControls addToast={addToast} />

            <LiveStats stats={stats} />

            <LiveTrafficChart stats={stats} />

            <LiveHistory history={history} addToast={addToast} />

            {/* Custom live dashboard toast notifications list */}
            <div className="toast-container toast-msg-container">
                {toasts.map(t => (
                    <div key={t.id} className={`toast show ${t.type}`}>
                        {t.type === "success" && (
                            <svg className="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                                <polyline points="22 4 12 14.01 9 11.01" />
                            </svg>
                        )}
                        {t.type === "warning" && (
                            <svg className="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                            </svg>
                        )}
                        {t.type === "error" && (
                            <svg className="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2" />
                                <line x1="12" y1="8" x2="12" y2="12" />
                                <line x1="12" y1="16" x2="12.01" y2="16" />
                            </svg>
                        )}
                        {t.type === "info" && (
                            <svg className="toast-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <circle cx="12" cy="12" r="10" />
                                <line x1="12" y1="16" x2="12" y2="12" />
                                <line x1="12" y1="8" x2="12.01" y2="8" />
                            </svg>
                        )}
                        <span className="toast-message">{t.message}</span>
                        <div className="toast-progress"></div>
                    </div>
                ))}
            </div>

        </div>

    );

}