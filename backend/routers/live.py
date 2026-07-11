from fastapi import APIRouter, Response
import threading
import io
import csv
import json

from backend.live.capture import (
    start_capture,
    stop_capture,
    is_capture_running
)

from backend.database import (
    get_recent_flows,
    get_statistics
)

from backend.live.flow_manager import flows

router = APIRouter()

capture_thread = None


# ==========================================
# START LIVE CAPTURE
# ==========================================

@router.post("/live/start")
def start_live_capture():

    global capture_thread

    # Already running
    if capture_thread and capture_thread.is_alive():

        return {
            "message": "Live capture is already running."
        }

    capture_thread = threading.Thread(
        target=start_capture,
        daemon=True
    )

    capture_thread.start()

    return {
        "message": "Live packet capture started."
    }


# ==========================================
# STOP LIVE CAPTURE
# ==========================================

@router.post("/live/stop")
def stop_live_capture():

    global capture_thread

    if not is_capture_running():

        return {
            "message": "Capture is not running."
        }

    stop_capture()

    if capture_thread:
        capture_thread = None

    return {
        "message": "Live packet capture stopped."
    }


# ==========================================
# CAPTURE STATUS
# ==========================================

@router.get("/live/status")
def capture_status():

    return {
        "running": is_capture_running()
    }


# ==========================================
# LIVE HISTORY
# ==========================================

@router.get("/live/history")
def live_history():

    return get_recent_flows(50)


# ==========================================
# LIVE STATISTICS
# ==========================================

@router.get("/live/stats")
def live_stats():

    stats = get_statistics()
    stats["active_flows"] = len(flows)
    return stats


# ==========================================
# DOWNLOAD LIVE HISTORY
# ==========================================

@router.get("/live/download/csv")
def download_live_csv():
    flows_list = get_recent_flows(10000)
    if not flows_list:
        # Return empty CSV headers
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["id", "flow_id", "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "start_time", "end_time", "duration", "packets", "bytes", "packets_per_second", "bytes_per_second", "prediction", "confidence", "created_at"])
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=live_history.csv"})

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=flows_list[0].keys())
    writer.writeheader()
    writer.writerows(flows_list)
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=live_history.csv",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )


@router.get("/live/download/json")
def download_live_json():
    flows_list = get_recent_flows(10000)
    return Response(
        content=json.dumps(flows_list, indent=4),
        media_type="application/json",
        headers={
            "Content-Disposition": "attachment; filename=live_history.json",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )