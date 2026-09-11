/*
 * CSV export.
 *
 * This used to be an <a href="http://127.0.0.1:8000/download">, pointing at a
 * server endpoint that streamed `predicted_output.csv` off the API's working
 * directory.  Two things were wrong with that:
 *
 *   1. The backend stopped writing that file (it was a race between concurrent
 *      requests, plus an unbounded disk write per request).  So the button
 *      downloaded a stale artifact from July no matter what you had uploaded.
 *      A silently wrong answer, not an error.
 *   2. Even when it did work, it was per-request state kept on the server's
 *      disk under a fixed filename -- two users uploading at once overwrite
 *      each other.
 *
 * The client already holds the full result. Build the CSV here. No round trip,
 * no shared mutable file, no staleness possible.
 */

function toCsv(result) {
    const rows = result?.predictions ?? [];
    const header = ["row", "prediction", "closed_set_confidence"];
    const body = rows.map((r, i) => [
        i + 1,
        // RFC-4180: quote, and escape embedded quotes by doubling them.
        `"${String(r.prediction).replace(/"/g, '""')}"`,
        Number(r.closed_set_confidence).toFixed(6),
    ]);
    return [header, ...body].map(cols => cols.join(",")).join("\n");
}

function DownloadButton({ result }) {
    if (!result) return null;

    const download = () => {
        const blob = new Blob([toCsv(result)], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `prediction_report_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);   // otherwise the blob leaks for the tab's lifetime
    };

    return (
        <div className="action-bar">
            <button
                className="cyber-btn"
                onClick={download}
                style={{
                    borderColor: "var(--neon-green)",
                    color: "var(--neon-green)",
                    boxShadow: "0 0 10px rgba(0, 230, 118, 0.1)",
                }}
            >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                Download Prediction Report (CSV)
            </button>
        </div>
    );
}

export default DownloadButton;
