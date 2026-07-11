import threading
import time
import logging

from backend.live.flow_manager import (
    flows,
    flows_lock,
    live_stats
)

from backend.live.extractor import extract_features
from backend.services.model_service import predict
from backend.database import insert_flow


# ==========================================
# Logger
# ==========================================

logger = logging.getLogger("cleanup_worker")

logging.basicConfig(level=logging.INFO)


# ==========================================
# Timeouts
# ==========================================

IDLE_TIMEOUT = 5.0
ACTIVE_TIMEOUT = 120.0


# ==========================================
# Cleanup Worker
# ==========================================

def cleanup_worker():

    while True:

        try:

            current = time.time()

            expired = []

            # -----------------------------
            # Collect expired flows
            # -----------------------------
            with flows_lock:

                for flow_id, flow in list(flows.items()):

                    idle_expired = (

                        current - flow.last_seen

                    ) > IDLE_TIMEOUT

                    active_expired = (

                        current - flow.first_seen

                    ) > ACTIVE_TIMEOUT

                    if idle_expired or active_expired:

                        expired.append((flow_id, flow))

                        del flows[flow_id]

            # -----------------------------
            # Process expired flows
            # -----------------------------
            for flow_id, flow in expired:

                try:

                    features = extract_features(flow)

                    attack, confidence = predict(features)

                    print("\n===================================")

                    print("FLOW FINISHED")

                    print(flow.flow_id)

                    print(
                        f"Prediction : {attack}"
                    )

                    print(
                        f"Confidence : {confidence:.2f}"
                    )

                    print("===================================\n")

                    duration = max(

                        flow.last_seen - flow.first_seen,

                        0.000001

                    )

                    stats = {

                        "duration": duration,

                        "packets_per_second":

                            flow.packets / duration,

                        "bytes_per_second":

                            flow.bytes / duration

                    }

                    # -------------------------
                    # Save to SQLite
                    # -------------------------
                    insert_flow(

                        flow,

                        stats,

                        attack,

                        confidence

                    )

                    # -------------------------
                    # Live Counter
                    # -------------------------
                    live_stats["flows_finished"] += 1

                except Exception as err:

                    logger.error(

                        f"Error processing flow {flow_id}: {err}"

                    )

        except Exception as err:

            logger.error(

                f"Cleanup worker failed: {err}"

            )

        time.sleep(1)


# ==========================================
# Start Cleanup Thread
# ==========================================

def start_cleanup():

    thread = threading.Thread(

        target=cleanup_worker,

        daemon=True,

        name="CleanupWorker"

    )

    thread.start()

    logger.info("Cleanup Worker Started")