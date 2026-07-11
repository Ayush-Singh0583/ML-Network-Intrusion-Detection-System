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