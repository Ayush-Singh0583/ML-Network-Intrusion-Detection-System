import sqlite3
from pathlib import Path

DATABASE_PATH = Path("network_ids.db")


def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():

    conn = get_connection()

    cursor = conn.cursor()

    # ===========================
    # FLOWS TABLE
    # ===========================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS flows (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        flow_id TEXT UNIQUE,

        src_ip TEXT,
        dst_ip TEXT,

        src_port INTEGER,
        dst_port INTEGER,

        protocol INTEGER,

        start_time REAL,
        end_time REAL,

        duration REAL,

        packets INTEGER,
        bytes INTEGER,

        packets_per_second REAL,
        bytes_per_second REAL,

        prediction TEXT,

        confidence REAL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ===========================
    # PACKETS TABLE
    # ===========================

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS packets(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        flow_id TEXT,

        timestamp REAL,

        packet_length INTEGER,

        tcp_flags TEXT,

        direction TEXT
    )
    """)

    # ===========================
    # SCAN ALERTS TABLE
    # ===========================
    #
    # Deliberately NOT a row in `flows`. A scan alert is a statement about a
    # SOURCE over a WINDOW: it has no flow_id, no duration and no packet
    # count, and the flow that triggered it is not the finding -- the pattern
    # across hundreds of flows is. Forcing it into the per-flow predictions
    # would mean inventing values for columns that have no meaning here, and
    # would make "how many scans today" a query over a column that mixes two
    # different kinds of judgement.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS scan_alerts (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        kind TEXT,                  -- 'vertical' | 'horizontal'

        src_ip TEXT,

        distinct_count INTEGER,     -- distinct ports (vertical) or hosts (horizontal)
        threshold INTEGER,

        window_seconds REAL,

        observations INTEGER,       -- events from this source still in the window

        first_seen REAL,
        last_seen REAL,

        sample TEXT,                -- a few example ports/hosts, for the analyst

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # The dashboard reads the newest alerts and counts per source; without
    # these every poll is a full scan of the table.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_alerts_id ON scan_alerts(id DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_alerts_src ON scan_alerts(src_ip)")

    conn.commit()

    conn.close()

    print("SQLite Database Initialized")
def insert_flow(flow, stats, prediction, confidence):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""

    INSERT OR REPLACE INTO flows(

        flow_id,

        src_ip,
        dst_ip,

        src_port,
        dst_port,

        protocol,

        start_time,
        end_time,

        duration,

        packets,
        bytes,

        packets_per_second,
        bytes_per_second,

        prediction,
        confidence

    )

    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)

    """, (

        flow.flow_id,

        flow.src_ip,
        flow.dst_ip,

        flow.src_port,
        flow.dst_port,

        flow.protocol,

        flow.first_seen,
        flow.last_seen,

        stats["duration"],

        flow.packets,
        flow.bytes,

        stats["packets_per_second"],
        stats["bytes_per_second"],

        prediction,
        confidence

    ))

    conn.commit()

    conn.close()
# ==========================================
# GET LATEST FLOWS
# ==========================================

def get_recent_flows(limit=100):

    conn = get_connection()

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    cursor.execute("""

        SELECT *

        FROM flows

        ORDER BY id DESC

        LIMIT ?

    """, (limit,))

    rows = cursor.fetchall()

    conn.close()

    return [dict(row) for row in rows]


# ==========================================
# LIVE STATISTICS
# ==========================================
# ==========================================
# LIVE STATISTICS
# ==========================================

def get_statistics():

    conn = get_connection()

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    # -----------------------------
    # Overall Statistics
    # -----------------------------
    cursor.execute("""

        SELECT

            COUNT(*)                     AS total_flows,

            COALESCE(SUM(packets),0)     AS total_packets,

            COALESCE(SUM(bytes),0)       AS total_bytes,

            COALESCE(AVG(packets_per_second),0) AS avg_pps,

            COALESCE(AVG(bytes_per_second),0)   AS avg_bps

        FROM flows

        WHERE prediction IS NOT NULL

    """)

    overall = dict(cursor.fetchone())

    # -----------------------------
    # Prediction Counts
    # -----------------------------
    cursor.execute("""

        SELECT

            prediction,

            COUNT(*) AS count

        FROM flows

        WHERE prediction IS NOT NULL

        GROUP BY prediction

    """)

    rows = cursor.fetchall()

    stats = {

        "total_flows": overall["total_flows"],

        "total_packets": overall["total_packets"],

        "total_bytes": overall["total_bytes"],

        "avg_packets_per_second": round(overall["avg_pps"], 2),

        "avg_bytes_per_second": round(overall["avg_bps"], 2)

    }

    for row in rows:

        stats[row["prediction"]] = row["count"]

    # -----------------------------
    # Protocol Counts
    # -----------------------------
    cursor.execute("""
        SELECT protocol, COUNT(*) AS count
        FROM flows
        GROUP BY protocol
    """)
    
    proto_rows = cursor.fetchall()
    protocols = {}
    for r in proto_rows:
        p_num = r["protocol"]
        p_name = "TCP" if p_num == 6 else ("UDP" if p_num == 17 else ("ICMP" if p_num == 1 else f"Other ({p_num})"))
        protocols[p_name] = r["count"]
        
    stats["protocols"] = protocols

    conn.close()

    return stats


# ==========================================
# SCAN ALERTS
# ==========================================


def insert_scan_alerts(alerts):
    """Persist a batch of ScanAlert objects.

    Batched on purpose: the cleanup worker drains a list once per tick, and
    one transaction for the batch beats one per alert. During an actual scan
    that is the difference between a handful of commits and thousands.
    """
    if not alerts:
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(
        """
        INSERT INTO scan_alerts(
            kind, src_ip, distinct_count, threshold, window_seconds,
            observations, first_seen, last_seen, sample
        ) VALUES (?,?,?,?,?,?,?,?,?)
        """,
        [
            (
                a.kind, a.src_ip, a.distinct, a.threshold, a.window_seconds,
                a.observations, a.first_seen, a.last_seen, ",".join(a.sample),
            )
            for a in alerts
        ],
    )
    conn.commit()
    conn.close()
    return len(alerts)


def get_recent_scan_alerts(limit=100):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM scan_alerts ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_scan_alert_summary():
    """Counts by kind and the worst offenders, for the dashboard header."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT kind, COUNT(*) AS n FROM scan_alerts GROUP BY kind")
    by_kind = {r["kind"]: r["n"] for r in cursor.fetchall()}

    cursor.execute(
        """
        SELECT src_ip,
               COUNT(*)            AS alerts,
               MAX(distinct_count) AS peak_distinct,
               MAX(last_seen)      AS last_seen
        FROM scan_alerts
        GROUP BY src_ip
        ORDER BY alerts DESC, peak_distinct DESC
        LIMIT 10
        """
    )
    top = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return {"total": sum(by_kind.values()), "by_kind": by_kind, "top_sources": top}
