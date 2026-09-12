import { useEffect, useState } from "react";
import axios from "axios";

/*
 * Model diagnostics, read from the API.
 *
 * This component used to hardcode:
 *
 *     Validated Accuracy   99.83%
 *     Extracted Features   78 Dimensions
 *     Target Classifications   15 Categories
 *
 * None of those came from an artifact in this repository. 99.83% was a
 * closed-set random-split figure typed in by hand; the loaded model has 60
 * features and 12 classes. A UI that states a metric it does not read is a UI
 * that will be wrong the first time the model changes -- and it was.
 *
 * It now reads GET /model, so the panel cannot drift from what is loaded. It
 * also reports DETECTION RATE AT A FIXED FALSE-ALARM BUDGET rather than
 * accuracy: on this dataset accuracy is ambiguous (58.47% multi-class vs
 * 69.68% binary against a 58.91% all-benign baseline) and says nothing about
 * whether the thing works.
 */

const API = "http://127.0.0.1:8000";
const pct = (x) => (x == null ? "—" : `${(x * 100).toFixed(2)}%`);

export default function ModelInfo() {
    const [info, setInfo] = useState(null);
    const [err, setErr] = useState(null);

    useEffect(() => {
        let alive = true;
        axios.get(`${API}/model`)
            .then((r) => alive && setInfo(r.data))
            .catch((e) => alive && setErr(e.message));
        return () => { alive = false; };
    }, []);

    const Metric = ({ label, value, cls = "" }) => (
        <div className="model-metric">
            <span className="model-metric-label">{label}</span>
            <span className={`model-metric-value ${cls}`}>{value}</span>
        </div>
    );

    return (
        <div className="model-card card">
            <h2>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ color: "var(--neon-cyan)" }}>
                    <circle cx="12" cy="12" r="3" />
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
                </svg>
                Model Intelligence &amp; Diagnostics
            </h2>

            {err && <p className="model-warn">Model API unreachable: {err}</p>}
            {!err && !info && <p className="model-warn">Loading model info…</p>}

            {info && !info.available && (
                <p className="model-warn">
                    No model loaded — <code>{info.bundle}</code>: {info.error}
                </p>
            )}

            {info && info.available && (
                <>
                    {info.smoke_test && (
                        <p className="model-warn">
                            ⚠ Smoke-test bundle — reduced training budget, not a reportable result.
                            {info.note ? ` (${info.note})` : ""}
                        </p>
                    )}

                    <div className="model-grid">
                        <Metric label="Classifier Engine" value={info.model} cls="active" />
                        <Metric label="Bundle" value={info.bundle} />
                        <Metric label="Split Protocol" value={info.protocol ?? "—"} />
                        <Metric label="Extracted Features" value={`${info.n_features} Dimensions`} />
                        <Metric label="Known Classes" value={`${info.classes?.length ?? 0} Categories`} />
                        <Metric
                            label="Novelty Rejector"
                            value={info.has_rejector ? `${info.scorer ?? "enabled"}` : "none"}
                            cls={info.has_rejector ? "active" : ""}
                        />
                    </div>

                    {info.metrics ? (
                        <div className="model-detection">
                            <h3>
                                Detection rate @ {(info.metrics.fpr_budget * 100).toFixed(1)}% false-alarm budget
                                <small> (observed {pct(info.metrics.observed_benign_fpr)} on the held-out test day)</small>
                            </h3>
                            <table className="detect-table">
                                <tbody>
                                    {Object.entries(info.metrics.per_class).map(([cls, rate]) => (
                                        <tr key={cls}>
                                            <td>{cls}</td>
                                            <td className={rate >= 0.5 ? "high-acc" : "low-acc"}>{pct(rate)}</td>
                                        </tr>
                                    ))}
                                    <tr className="detect-total">
                                        <td>ANY ATTACK</td>
                                        <td>{pct(info.metrics.any_attack)}</td>
                                    </tr>
                                </tbody>
                            </table>
                            <p className="detect-note">
                                Detection rate at a fixed false-alarm budget, not accuracy — accuracy on this
                                split is ambiguous (58.47% multi-class vs 69.68% binary, against a 58.91%
                                all-benign baseline).
                            </p>
                        </div>
                    ) : (
                        <p className="model-warn">
                            This bundle carries no evaluation metrics. Train via
                            <code> python src/run.py train --model rf --protocol crossday</code>.
                        </p>
                    )}
                </>
            )}
        </div>
    );
}
