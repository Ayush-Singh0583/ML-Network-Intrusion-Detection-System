---
title: "Database Schema and Persistence Layer"
category: "Backend"
sources:
  - "raw/PROJECT_DOCUMENTATION.md"
tags: ["sqlite", "database", "schema", "persistence", "telemetry"]
updated: 2026-08-26
---

# Database Schema & Persistence Layer

## 📌 Executive Summary
The ML-NIDS persists all classified network flows into an **SQLite** database (`network_ids.db`) managed via `backend/database.py`. The database supports real-time telemetry insertion, dynamic statistical queries for the React SOC frontend, and historical CSV logging exports.

---

## 🗄️ Table Schema: `flows`

The primary data store for all completed network sessions:

| Column Name | Data Type | Constraint / Default | Description |
| :--- | :--- | :--- | :--- |
| `id` | `INTEGER` | `PRIMARY KEY AUTOINCREMENT` | Unique row identifier. |
| `flow_id` | `TEXT` | `NOT NULL` | Symmetrical 5-tuple hash identifier. |
| `src_ip` | `TEXT` | `NOT NULL` | Source IP address of flow initiator. |
| `dst_ip` | `TEXT` | `NOT NULL` | Destination IP address of target host. |
| `src_port` | `INTEGER` | `NOT NULL` | Source port number. |
| `dst_port` | `INTEGER` | `NOT NULL` | Destination port number. |
| `protocol` | `INTEGER` | `NOT NULL` | IP protocol number (e.g. 6 = TCP, 17 = UDP, 1 = ICMP). |
| `start_time` | `REAL` | `NOT NULL` | Epoch timestamp of first observed packet. |
| `end_time` | `REAL` | `NOT NULL` | Epoch timestamp of last observed packet. |
| `duration` | `REAL` | `NOT NULL` | Session duration in seconds. |
| `packets` | `INTEGER` | `NOT NULL` | Total packets aggregated in session. |
| `bytes` | `INTEGER` | `NOT NULL` | Total payload + header bytes aggregated. |
| `packets_per_second` | `REAL` | `NOT NULL` | Mean packet throughput rate. |
| `bytes_per_second` | `REAL` | `NOT NULL` | Mean byte throughput rate. |
| `prediction` | `TEXT` | `NOT NULL` | Classification label (e.g. `BENIGN`, `DDoS`, `Unknown_Attack`). |
| `confidence` | `REAL` | `NOT NULL` | Maximum posterior probability score ($0.0$ to $1.0$). |
| `created_at` | `TIMESTAMP` | `DEFAULT CURRENT_TIMESTAMP` | Row insertion timestamp. |

---

## 🔍 Database Operations (`backend/database.py`)
1. `insert_flow(...)`: Inserts completed flow telemetry and classification score.
2. `get_total_flows()` & `get_total_packets()`: Computes aggregate throughput counters for the dashboard summary cards.
3. `get_threat_counts()`: Aggregates detection frequencies grouped by `prediction` label for donut and bar chart visualization.
4. `get_recent_flows(limit, offset, search)`: Returns paginated flow history with text filtering across IP and attack labels.
5. `export_flows_csv()`: Dumps database records into CSV for offline analysis.

---

## 🔗 Related Topics (Wikilinks)
- [[Live-Packet-Flow-Pipeline]] — Producer of flow records inserted into SQLite.
- [[System-Architecture]] — FastAPI query integration and React dashboard consumer.
- [[Open-Set-Recognition]] — Persisting `Unknown_Attack` alerts.
- [[Index]] — Master Knowledge Graph Index.
