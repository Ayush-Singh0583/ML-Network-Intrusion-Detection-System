"""
Synthetic CIC-IDS2017-shaped captures, for tests and smoke runs.

WHAT THIS IS FOR
----------------
The study cannot be developed against the real dataset in every environment
(863 MB of CSV, and the distribution with timestamps is a separate download).
This module writes eight small CSVs that reproduce the *shape* of the real
files -- including the parts of the shape that break naive code:

* the real header, leading spaces and all, with ``Fwd Header Length`` written
  twice (pandas renames the second to ``Fwd Header Length.1``);
* the GeneratedLabelledFlows layout (85 columns, with Flow ID, addresses,
  ports and Timestamp) or the MachineLearningCSV layout (79 columns, none of
  those except Destination Port);
* timestamps on a 12-hour clock with no AM/PM marker, minute resolution on
  every day but Monday;
* the Thursday-morning labels written with the single byte 0x96, and that
  file's trailing block of completely empty rows;
* literal ``Infinity`` / ``NaN`` in the rate columns, a few negative flow
  durations, and flows recorded with the server as their source.

WHAT THIS IS NOT
----------------
Evidence.  The feature values are drawn from distributions chosen so that the
known qualitative behaviours appear in miniature (a port scan is a burst of
tiny, near-identical flows from one source to many ports; a DDoS looks like
DoS Hulk; a bot beacon looks benign).  A number measured on this data says the
code runs.  It says nothing about intrusion detection, and no figure produced
from it may appear in the paper.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Raw header exactly as the GeneratedLabelledFlows CSVs ship it.  The second
# " Fwd Header Length" is the duplicate pandas renames to ".1".
GLF_HEADER: List[str] = [
    "Flow ID", " Source IP", " Source Port", " Destination IP", " Destination Port",
    " Protocol", " Timestamp", " Flow Duration", " Total Fwd Packets",
    " Total Backward Packets", "Total Length of Fwd Packets",
    " Total Length of Bwd Packets", " Fwd Packet Length Max", " Fwd Packet Length Min",
    " Fwd Packet Length Mean", " Fwd Packet Length Std", "Bwd Packet Length Max",
    " Bwd Packet Length Min", " Bwd Packet Length Mean", " Bwd Packet Length Std",
    "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std",
    " Flow IAT Max", " Flow IAT Min", "Fwd IAT Total", " Fwd IAT Mean", " Fwd IAT Std",
    " Fwd IAT Max", " Fwd IAT Min", "Bwd IAT Total", " Bwd IAT Mean", " Bwd IAT Std",
    " Bwd IAT Max", " Bwd IAT Min", "Fwd PSH Flags", " Bwd PSH Flags", " Fwd URG Flags",
    " Bwd URG Flags", " Fwd Header Length", " Bwd Header Length", "Fwd Packets/s",
    " Bwd Packets/s", " Min Packet Length", " Max Packet Length", " Packet Length Mean",
    " Packet Length Std", " Packet Length Variance", "FIN Flag Count", " SYN Flag Count",
    " RST Flag Count", " PSH Flag Count", " ACK Flag Count", " URG Flag Count",
    " CWE Flag Count", " ECE Flag Count", " Down/Up Ratio", " Average Packet Size",
    " Avg Fwd Segment Size", " Avg Bwd Segment Size", " Fwd Header Length",
    "Fwd Avg Bytes/Bulk", " Fwd Avg Packets/Bulk", " Fwd Avg Bulk Rate",
    " Bwd Avg Bytes/Bulk", " Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", " Subflow Fwd Bytes", " Subflow Bwd Packets",
    " Subflow Bwd Bytes", "Init_Win_bytes_forward", " Init_Win_bytes_backward",
    " act_data_pkt_fwd", " min_seg_size_forward", "Active Mean", " Active Std",
    " Active Max", " Active Min", "Idle Mean", " Idle Std", " Idle Max", " Idle Min",
    " Label",
]
_IDENT = ["Flow ID", " Source IP", " Source Port", " Destination IP", " Protocol", " Timestamp"]
MLCSV_HEADER: List[str] = [h for h in GLF_HEADER if h not in _IDENT]

DAY_DATE = {"Monday": (3, 7), "Tuesday": (4, 7), "Wednesday": (5, 7),
            "Thursday": (6, 7), "Friday": (7, 7)}

# file -> (day, start hour, end hour, [(label, relative count, start frac, end frac)])
# The fractions place each attack inside its capture window, as in the real week.
CAPTURES: Dict[str, tuple] = {
    "Monday-WorkingHours.pcap_ISCX.csv": ("Monday", 9.0, 17.0, []),
    "Tuesday-WorkingHours.pcap_ISCX.csv": ("Tuesday", 9.0, 17.0, [
        ("FTP-Patator", 0.30, 0.05, 0.18), ("SSH-Patator", 0.25, 0.62, 0.75)]),
    "Wednesday-workingHours.pcap_ISCX.csv": ("Wednesday", 9.0, 17.0, [
        ("DoS slowloris", 0.20, 0.10, 0.14), ("DoS Slowhttptest", 0.20, 0.16, 0.20),
        ("DoS Hulk", 0.90, 0.22, 0.27), ("DoS GoldenEye", 0.30, 0.28, 0.31),
        ("Heartbleed", 0.012, 0.78, 0.80)]),
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": ("Thursday", 9.0, 12.5, [
        ("Web Attack \u2013 Brute Force", 0.25, 0.10, 0.40),
        ("Web Attack \u2013 XSS", 0.15, 0.45, 0.60),
        ("Web Attack \u2013 Sql Injection", 0.02, 0.62, 0.66)]),
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": ("Thursday", 12.5, 17.0, [
        ("Infiltration", 0.03, 0.30, 0.60)]),
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": ("Friday", 9.0, 12.5, [
        ("Bot", 0.20, 0.30, 0.60)]),
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": ("Friday", 12.5, 15.5, [
        ("PortScan", 1.20, 0.45, 0.47)]),      # a burst: see SCAN_FLOWS_PER_MINUTE
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": ("Friday", 15.5, 17.0, [
        ("DDoS", 1.00, 0.30, 0.55)]),
}

# Scans are placed at a fixed RATE, starting on a minute boundary, so that a
# small dataset holds a short scan rather than a slow one: the behaviour layer
# counts ports per source per minute, and a scan thinned out to fit a small
# flow count would stop being a scan.
SCAN_FLOWS_PER_MINUTE = 240


def _scan_times(start_seconds: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t0 = np.floor(start_seconds / 60.0) * 60.0
    return np.sort(t0 + rng.uniform(0.0, 60.0 * n / SCAN_FLOWS_PER_MINUTE, n))


CLIENTS = [f"192.168.10.{i}" for i in (5, 8, 9, 12, 14, 15, 16, 17, 19, 25)]
SERVERS = ["104.16.207.165", "104.16.28.216", "172.217.10.46", "23.60.139.27",
           "13.107.4.50", "151.101.1.69", "192.168.10.3"]
WEB_SERVER = "192.168.10.50"
ATTACKER = "172.16.0.1"


def _stamp(day: str, seconds_of_day: float, with_seconds: bool) -> str:
    """Timestamp string the way the dataset prints it: day first, 12-hour clock,
    no AM/PM, minute resolution unless ``with_seconds``."""
    d, m = DAY_DATE[day]
    sec = int(seconds_of_day)
    h24, mi, s = sec // 3600, (sec % 3600) // 60, sec % 60
    h12 = h24 if h24 <= 12 else h24 - 12
    if with_seconds:
        return f"{d:02d}/{m:02d}/2017 {h12:02d}:{mi:02d}:{s:02d}"
    return f"{d}/{m}/2017 {h12}:{mi:02d}"


def _features(label: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Class-conditional flow features (stripped names).  See module docstring:
    plausible in shape, not a measurement of anything."""
    base = label.replace("\u2013", "-")
    f: Dict[str, np.ndarray] = {}

    def pos(mean, sd):
        return np.abs(rng.normal(mean, sd, n))

    # ---- geometry per class ------------------------------------------------
    if base == "PortScan":
        # one SYN out, one RST back, no payload: thousands of near-copies
        fwd_pk = rng.choice([1, 2], n, p=[0.85, 0.15]).astype(float)
        bwd_pk = rng.choice([0, 1], n, p=[0.1, 0.9]).astype(float)
        dur = rng.choice([30.0, 45.0, 60.0, 80.0], n)
        fwd_len = np.zeros(n); bwd_len = rng.choice([0.0, 6.0], n, p=[0.5, 0.5])
        win_f = np.full(n, 1024.0); win_b = np.zeros(n)
        syn = np.ones(n); psh = np.zeros(n)
    elif base in ("DoS Hulk", "DDoS"):
        # volumetric HTTP floods share a geometry -- that is the point
        shift = 0.0 if base == "DoS Hulk" else 0.25
        fwd_pk = np.round(pos(6 + shift, 1.5)) + 1
        bwd_pk = np.round(pos(5 + shift, 1.5)) + 1
        dur = pos(8.5e6 * (1 + shift), 2.0e6)
        fwd_len = pos(350, 60); bwd_len = pos(11600, 900)
        win_f = rng.choice([256.0, 8192.0], n); win_b = np.full(n, 229.0)
        syn = np.zeros(n); psh = np.ones(n)
    elif base in ("DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"):
        k = {"DoS GoldenEye": 1.0, "DoS slowloris": 2.0, "DoS Slowhttptest": 3.0}[base]
        fwd_pk = np.round(pos(4 + k, 1.0)) + 1
        bwd_pk = np.round(pos(2 + k, 1.0))
        dur = pos(3.0e7 * k, 5.0e6)
        fwd_len = pos(200 * k, 40); bwd_len = pos(900 * k, 200)
        win_f = np.full(n, 29200.0); win_b = rng.choice([235.0, 28960.0], n)
        syn = np.zeros(n); psh = np.ones(n)
    elif base in ("FTP-Patator", "SSH-Patator"):
        k = 1.0 if base == "FTP-Patator" else 2.2
        fwd_pk = np.round(pos(8 * k, 1.5)) + 2
        bwd_pk = np.round(pos(9 * k, 1.5)) + 2
        dur = pos(4.0e6 * k, 6.0e5)
        fwd_len = pos(70 * k, 12); bwd_len = pos(140 * k, 25)
        win_f = np.full(n, 29200.0); win_b = np.full(n, 227.0 if k == 1.0 else 247.0)
        syn = np.zeros(n); psh = np.ones(n)
    elif base.startswith("Web Attack"):
        k = {"Web Attack - Brute Force": 1.0, "Web Attack - XSS": 1.6,
             "Web Attack - Sql Injection": 2.4}[base]
        fwd_pk = np.round(pos(3 * k, 0.8)) + 1
        bwd_pk = np.round(pos(2 * k, 0.8)) + 1
        dur = pos(5.5e6 * k, 4.0e5)
        fwd_len = pos(600 * k, 60); bwd_len = pos(1500 * k, 150)
        win_f = np.full(n, 29200.0); win_b = np.full(n, 28960.0)
        syn = np.zeros(n); psh = np.ones(n)
    elif base == "Heartbleed":
        fwd_pk = np.round(pos(2700, 80)); bwd_pk = np.round(pos(2000, 80))
        dur = pos(1.19e8, 2.0e5)
        fwd_len = pos(8000, 300); bwd_len = pos(7.8e6, 1.0e5)
        win_f = np.full(n, 29200.0); win_b = np.full(n, 235.0)
        syn = np.zeros(n); psh = np.ones(n)
    else:
        # BENIGN, and the two classes that are benign-looking per flow
        # (Bot, Infiltration): a mixture of short lookups, web sessions and
        # failed connections that resemble nothing in particular
        kind = rng.choice(3, n, p=[0.35, 0.5, 0.15])
        fwd_pk = np.where(kind == 0, 1 + rng.integers(0, 2, n),
                          np.where(kind == 1, np.round(pos(9, 5)) + 2, 1 + rng.integers(0, 3, n))).astype(float)
        bwd_pk = np.where(kind == 0, 1 + rng.integers(0, 2, n),
                          np.where(kind == 1, np.round(pos(10, 6)) + 1, rng.integers(0, 2, n))).astype(float)
        dur = np.where(kind == 0, pos(3.0e4, 2.0e4),
                       np.where(kind == 1, pos(2.5e6, 2.0e6), pos(1.2e3, 1.0e3)))
        fwd_len = np.where(kind == 0, pos(70, 25), np.where(kind == 1, pos(900, 600), pos(8, 8)))
        bwd_len = np.where(kind == 0, pos(160, 60), np.where(kind == 1, pos(5200, 4000), pos(8, 8)))
        win_f = rng.choice([8192.0, 29200.0, 65535.0, 251.0, -1.0], n, p=[0.3, 0.3, 0.2, 0.1, 0.1])
        win_b = rng.choice([229.0, 28960.0, 65535.0, -1.0], n, p=[0.3, 0.3, 0.2, 0.2])
        syn = (kind == 2).astype(float); psh = (kind == 1).astype(float)
        if base == "Bot":
            dur = dur * 1.05
        if base == "Infiltration":
            bwd_len = bwd_len * 1.1

    dur = np.maximum(np.round(dur), 0.0)
    tot_pk = fwd_pk + bwd_pk
    with np.errstate(divide="ignore", invalid="ignore"):
        bytes_s = (fwd_len + bwd_len) / (dur / 1e6)
        pk_s = tot_pk / (dur / 1e6)

    fwd_mean = np.divide(fwd_len, fwd_pk, out=np.zeros(n), where=fwd_pk > 0)
    bwd_mean = np.divide(bwd_len, bwd_pk, out=np.zeros(n), where=bwd_pk > 0)
    iat = np.divide(dur, np.maximum(tot_pk - 1, 1))
    hdr_f = 20.0 * fwd_pk + rng.choice([0.0, 12.0], n)
    hdr_b = 20.0 * bwd_pk

    f["Flow Duration"] = dur
    f["Total Fwd Packets"] = fwd_pk
    f["Total Backward Packets"] = bwd_pk
    f["Total Length of Fwd Packets"] = np.round(fwd_len)
    f["Total Length of Bwd Packets"] = np.round(bwd_len)
    f["Fwd Packet Length Max"] = np.round(fwd_mean * 1.6)
    f["Fwd Packet Length Min"] = np.round(fwd_mean * 0.3)
    f["Fwd Packet Length Mean"] = fwd_mean
    f["Fwd Packet Length Std"] = fwd_mean * 0.4
    f["Bwd Packet Length Max"] = np.round(bwd_mean * 1.7)
    f["Bwd Packet Length Min"] = np.round(bwd_mean * 0.2)
    f["Bwd Packet Length Mean"] = bwd_mean
    f["Bwd Packet Length Std"] = bwd_mean * 0.5
    f["Flow Bytes/s"] = bytes_s
    f["Flow Packets/s"] = pk_s
    f["Flow IAT Mean"] = iat
    f["Flow IAT Std"] = iat * rng.uniform(0.2, 1.2, n)
    f["Flow IAT Max"] = iat * rng.uniform(1.0, 3.0, n)
    f["Flow IAT Min"] = iat * rng.uniform(0.0, 0.5, n)
    f["Fwd IAT Total"] = dur * 0.9
    f["Fwd IAT Mean"] = iat * 1.8
    f["Fwd IAT Std"] = iat * 0.9
    f["Fwd IAT Max"] = iat * 3.2
    f["Fwd IAT Min"] = iat * 0.2
    f["Bwd IAT Total"] = dur * 0.8
    f["Bwd IAT Mean"] = iat * 1.7
    f["Bwd IAT Std"] = iat * 0.8
    f["Bwd IAT Max"] = iat * 3.0
    f["Bwd IAT Min"] = iat * 0.25
    f["Fwd PSH Flags"] = (psh * (rng.random(n) < 0.3)).astype(float)
    f["Bwd PSH Flags"] = np.zeros(n)          # constant in the real data too
    f["Fwd URG Flags"] = np.zeros(n)
    f["Bwd URG Flags"] = np.zeros(n)
    f["Fwd Header Length"] = hdr_f
    f["Bwd Header Length"] = hdr_b
    f["Fwd Packets/s"] = pk_s * np.divide(fwd_pk, np.maximum(tot_pk, 1))
    f["Bwd Packets/s"] = pk_s * np.divide(bwd_pk, np.maximum(tot_pk, 1))
    f["Min Packet Length"] = np.minimum(f["Fwd Packet Length Min"], f["Bwd Packet Length Min"])
    f["Max Packet Length"] = np.maximum(f["Fwd Packet Length Max"], f["Bwd Packet Length Max"])
    f["Packet Length Mean"] = np.divide(fwd_len + bwd_len, np.maximum(tot_pk, 1))
    f["Packet Length Std"] = f["Packet Length Mean"] * 0.6
    f["Packet Length Variance"] = f["Packet Length Std"] ** 2
    f["FIN Flag Count"] = (rng.random(n) < 0.3).astype(float)
    f["SYN Flag Count"] = syn
    f["RST Flag Count"] = np.zeros(n)
    f["PSH Flag Count"] = psh
    f["ACK Flag Count"] = (bwd_pk > 0).astype(float)
    f["URG Flag Count"] = (rng.random(n) < 0.05).astype(float)
    f["CWE Flag Count"] = np.zeros(n)
    f["ECE Flag Count"] = np.zeros(n)
    f["Down/Up Ratio"] = np.floor(np.divide(bwd_pk, np.maximum(fwd_pk, 1)))
    f["Average Packet Size"] = f["Packet Length Mean"] * 1.1
    f["Avg Fwd Segment Size"] = fwd_mean                     # collinear twin
    f["Avg Bwd Segment Size"] = bwd_mean                     # collinear twin
    f["Fwd Header Length.1"] = hdr_f                         # the duplicated header
    for c in ("Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
              "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"):
        f[c] = np.zeros(n)
    f["Subflow Fwd Packets"] = fwd_pk                        # collinear twin
    f["Subflow Fwd Bytes"] = f["Total Length of Fwd Packets"]
    f["Subflow Bwd Packets"] = bwd_pk
    f["Subflow Bwd Bytes"] = f["Total Length of Bwd Packets"]
    f["Init_Win_bytes_forward"] = win_f
    f["Init_Win_bytes_backward"] = win_b
    f["act_data_pkt_fwd"] = np.maximum(fwd_pk - 1, 0)
    f["min_seg_size_forward"] = rng.choice([20.0, 32.0], n)
    active = np.where(dur > 1e6, dur * 0.1, 0.0)
    f["Active Mean"] = active; f["Active Std"] = active * 0.1
    f["Active Max"] = active * 1.2; f["Active Min"] = active * 0.8
    idle = np.where(dur > 5e6, dur * 0.6, 0.0)
    f["Idle Mean"] = idle; f["Idle Std"] = idle * 0.05
    f["Idle Max"] = idle * 1.1; f["Idle Min"] = idle * 0.9
    return pd.DataFrame(f)


def _endpoints(label: str, n: int, t: np.ndarray, rng: np.random.Generator):
    """(src_ip, src_port, dst_ip, dst_port, proto) per flow."""
    base = label.replace("\u2013", "-")
    src = np.empty(n, dtype=object); dst = np.empty(n, dtype=object)
    sport = rng.integers(32768, 61000, n); dport = np.zeros(n, dtype=np.int64)
    proto = np.full(n, 6, dtype=np.int64)

    if base == "PortScan":
        src[:] = ATTACKER; dst[:] = WEB_SERVER
        dport = rng.integers(1, 20000, n)            # many distinct ports
    elif base in ("DDoS", "DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest",
                  "Heartbleed") or base.startswith("Web Attack"):
        src[:] = ATTACKER; dst[:] = WEB_SERVER
        dport[:] = 444 if base == "Heartbleed" else 80
    elif base == "FTP-Patator":
        src[:] = ATTACKER; dst[:] = WEB_SERVER; dport[:] = 21
    elif base == "SSH-Patator":
        src[:] = ATTACKER; dst[:] = WEB_SERVER; dport[:] = 22
    elif base == "Bot":
        src[:] = rng.choice(CLIENTS[:5], n); dst[:] = "205.174.165.73"; dport[:] = 8080
    elif base == "Infiltration":
        src[:] = "192.168.10.8"; dst[:] = "205.174.165.73"; dport[:] = 444
    else:
        src[:] = rng.choice(CLIENTS, n)
        dst[:] = rng.choice(SERVERS, n)
        dport = rng.choice([443, 80, 53, 123, 22], n, p=[0.5, 0.25, 0.2, 0.03, 0.02])
        proto = np.where(np.isin(dport, [53, 123]), 17, 6)
        # ~12% of benign flows are recorded with the SERVER as source, as
        # CICFlowMeter does when its first packet of a flow is a reply
        rev = rng.random(n) < 0.12
        src[rev], dst[rev] = dst[rev], src[rev].copy()
        sport_r = sport.copy()
        sport = np.where(rev, dport, sport)
        dport = np.where(rev, sport_r, dport)
    return src, sport, dst, dport, proto


def make_capture(
    fname: str,
    n_benign: int = 1500,
    attack_scale: int = 500,
    seed: int = 0,
    distribution: str = "glf",
) -> pd.DataFrame:
    """One capture as a DataFrame whose columns are the RAW header strings."""
    if distribution not in ("glf", "mlcsv"):
        raise ValueError("distribution must be 'glf' or 'mlcsv'")
    day, h0, h1, attacks = CAPTURES[fname]
    # zlib.crc32, not hash(): str hashes are salted per process, which would
    # make the 'same' synthetic dataset different on every run
    rng = np.random.default_rng((zlib.crc32(fname.encode()) + 7919 * int(seed)) % (2 ** 32))
    span = (h1 - h0) * 3600.0

    parts = []
    plan = [("BENIGN", n_benign, 0.0, 1.0)] + [
        (lab, max(int(round(rel * attack_scale)), 6), a, b) for lab, rel, a, b in attacks
    ]
    for lab, n, a, b in plan:
        if lab == "PortScan":
            t = _scan_times(h0 * 3600.0 + span * a, n, rng)
        else:
            t = np.sort(h0 * 3600.0 + span * rng.uniform(a, b, n))
        feats = _features(lab, n, rng)
        src, sport, dst, dport, proto = _endpoints(lab, n, t, rng)
        feats["__t"] = t
        feats["__src"] = src; feats["__sport"] = sport
        feats["__dst"] = dst; feats["__dport"] = dport; feats["__proto"] = proto
        feats["__label"] = lab
        parts.append(feats)

    # the Thursday-afternoon capture also holds a scan from an infected host
    # that the original labels call BENIGN -- as the real one does
    if fname.startswith("Thursday-WorkingHours-Afternoon"):
        n = max(int(attack_scale * 0.8), 200)
        t = _scan_times(h0 * 3600.0 + span * 0.62, n, rng)
        feats = _features("PortScan", n, rng)
        feats["__t"] = t
        feats["__src"] = "192.168.10.8"; feats["__sport"] = rng.integers(32768, 61000, n)
        feats["__dst"] = rng.choice([f"192.168.10.{i}" for i in range(5, 26)], n)
        feats["__dport"] = rng.integers(1, 2000, n); feats["__proto"] = 6
        feats["__label"] = "BENIGN"
        parts.append(feats)

    df = pd.concat(parts, ignore_index=True).sort_values("__t", kind="stable").reset_index(drop=True)
    n = len(df)

    # ---- the format quirks -------------------------------------------------
    # the unique names: the duplicated header is 'Fwd Header Length.1' here
    stripped = [h.strip() for h in GLF_HEADER_UNIQUE]
    out = pd.DataFrame(index=df.index)
    out["Flow ID"] = [f"{a}-{b}-{c}-{d}-{e}" for a, b, c, d, e in
                      zip(df["__src"], df["__dst"], df["__sport"], df["__dport"], df["__proto"])]
    out["Source IP"] = df["__src"]; out["Source Port"] = df["__sport"]
    out["Destination IP"] = df["__dst"]; out["Destination Port"] = df["__dport"]
    out["Protocol"] = df["__proto"]
    out["Timestamp"] = [_stamp(day, s, with_seconds=(day == "Monday")) for s in df["__t"]]
    for c in stripped[7:-1]:
        out[c] = df[c]
    out["Label"] = df["__label"]

    # rate columns: the real files print Infinity and NaN for zero-duration flows
    for c in ("Flow Bytes/s", "Flow Packets/s"):
        v = out[c].to_numpy(dtype=np.float64)
        s = pd.Series(v, index=out.index).round(4).astype(object)
        s[np.isinf(v)] = "Infinity"
        s[np.isnan(v)] = "NaN"
        out[c] = s
    # a handful of negative durations, as CICFlowMeter emits on clock skew
    neg = rng.choice(n, size=min(3, n), replace=False)
    out.loc[neg, "Flow Duration"] = -1

    if distribution == "mlcsv":
        out = out.drop(columns=["Flow ID", "Source IP", "Source Port", "Destination IP",
                                "Protocol", "Timestamp"])
        # that distribution carries U+FFFD where the other carries byte 0x96
        # (an en dash in Windows-1252)
        out["Label"] = out["Label"].str.replace("\u2013", "\ufffd", regex=False)
        out.columns = MLCSV_HEADER_UNIQUE
    else:
        out.columns = GLF_HEADER_UNIQUE
    return out


def _unique(header: List[str]) -> List[str]:
    """Column names for the DataFrame: the duplicated header gets the ``.1``
    suffix pandas would add, so the frame is addressable; ``write_capture``
    writes the true duplicated header line."""
    seen, out = set(), []
    for h in header:
        out.append(h + ".1" if h in seen else h)
        seen.add(h)
    return out


GLF_HEADER_UNIQUE = _unique(GLF_HEADER)
MLCSV_HEADER_UNIQUE = _unique(MLCSV_HEADER)


def write_capture(df: pd.DataFrame, path: Path, distribution: str = "glf",
                  blank_rows: int = 0) -> None:
    """Write one capture the way the dataset ships it: the duplicated header
    line verbatim, cp1252 bytes for the GLF labels, optional trailing rows that
    are empty in every column."""
    header = GLF_HEADER if distribution == "glf" else MLCSV_HEADER
    body = df.to_csv(index=False, header=False, lineterminator="\n")
    text = ",".join(header) + "\n" + body
    if blank_rows:
        text += ("," * (len(header) - 1) + "\n") * blank_rows
    encoding = "cp1252" if distribution == "glf" else "utf-8"
    Path(path).write_bytes(text.encode(encoding))


def write_dataset(
    out_dir: Path,
    n_benign: int = 1500,
    attack_scale: int = 500,
    seed: int = 0,
    distribution: str = "glf",
    only: Optional[List[str]] = None,
) -> Path:
    """Write all eight captures into ``out_dir`` and return it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in CAPTURES:
        if only is not None and fname not in only:
            continue
        df = make_capture(fname, n_benign, attack_scale, seed, distribution)
        blanks = 40 if (distribution == "glf" and "Morning-WebAttacks" in fname) else 0
        write_capture(df, out_dir / fname, distribution, blank_rows=blanks)
    return out_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="write synthetic CIC-IDS2017-shaped CSVs")
    ap.add_argument("out_dir")
    ap.add_argument("--n-benign", type=int, default=1500)
    ap.add_argument("--attack-scale", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--distribution", choices=["glf", "mlcsv"], default="glf")
    a = ap.parse_args()
    p = write_dataset(Path(a.out_dir), a.n_benign, a.attack_scale, a.seed, a.distribution)
    print(f"wrote {len(CAPTURES)} synthetic captures to {p}  (NOT real data)")
