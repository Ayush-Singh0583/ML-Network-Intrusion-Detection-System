"""
Windowed scan detection.

WHY THIS EXISTS
---------------
The flow classifier cannot detect port scans, and that is structural rather
than a tuning failure.  A scan is the same flow repeated against different
ports: identical duration, packet counts, lengths and TCP flags.  Measured on
this project's own data, 158,930 PortScan flows collapse to **1,958 unique
feature vectors** once ``Destination Port`` is removed as a shortcut feature
(90,819 with it).  Four independent models -- a 25-tree Random Forest, a
300-tree Random Forest, a 400-round class-balanced XGBoost, and every decision
threshold between 0.1% and 5% false alarms -- all return 0.09%-0.26% detection.

The signal is not in any single flow, because **scanning is a property of a
SET of flows**: one source touching many ports, or many hosts, in a short
window.  So it is detected by counting, not by classifying, and it runs
alongside the model rather than inside it.

WHAT IT DETECTS
---------------
    vertical    one source  ->  many PORTS on few hosts   (nmap -p-)
    horizontal  one source  ->  many HOSTS on few ports   (subnet sweep for 445)

Catching only the first is the common mistake; CIC-IDS2017's PortScan class is
mostly vertical, which makes it easy to forget the other shape exists.

DESIGN NOTES
------------
*Observed at flow CREATION, not flow expiry.*  A SYN scan against closed ports
produces flows that receive one packet and never another, so they sit in the
flow table until ``IDLE_TIMEOUT`` (5s) fires.  Feeding the tracker from the
cleanup worker would therefore detect a scan roughly six seconds after it
finished, in one lump, having ignored every packet while it was happening.
The moment a new flow key appears IS the observation "this source just touched
this port" -- no waiting and no model involved.

*Bounded, because otherwise this is a memory-exhaustion vector.*  Keying by
source IP with no cap means a scanner using spoofed or decoy source addresses
(``nmap -D``, or any flood with randomised sources) allocates a new entry per
fake source.  A detector that can be made to exhaust the host's memory by
sending it traffic is an attack surface, not a control.  ``max_sources`` caps
the table and eviction is least-recently-observed.

*Its own lock, and it is always a leaf.*  ``ScanTracker`` acquires only
``self._lock`` and never calls back into the flow manager, the database or the
model service while holding it.  A lock that is always acquired last cannot
participate in a cycle, so no ordering discipline is needed of callers -- they
may hold ``flows_lock`` across a call into here safely.

*Time is injected.*  Every method takes ``now``.  That is what lets the whole
detector be tested with synthetic ``(src_ip, dst_ip, dst_port, timestamp)``
tuples: no packets, no sockets, no threads, no sleeping.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

# =====================================================================
# DEFAULTS
# ---------------------------------------------------------------------
# Thresholds are the part a deployment must tune; these are starting points
# chosen to sit well clear of ordinary client behaviour.  A browser opens many
# connections but nearly all to port 443 on a handful of hosts.  The usual
# false positives are NAT gateways, proxies, load balancers and vulnerability
# scanners you run yourself -- put those in ``ignore_sources``.
# =====================================================================

WINDOW_SECONDS = 60.0
VERTICAL_PORT_THRESHOLD = 100     # distinct destination ports from one source
HORIZONTAL_HOST_THRESHOLD = 50    # distinct destination hosts from one source
MAX_SOURCES = 50_000              # cap on tracked source addresses
ALERT_COOLDOWN_SECONDS = 60.0     # per source, per scan kind
SAMPLE_SIZE = 10                  # examples carried on an alert, for the analyst


@dataclass(frozen=True)
class ScanAlert:
    """One scan observation about a SOURCE over a WINDOW.

    Deliberately not shaped like a flow.  It has no flow_id, no duration and
    no packet count, because it is not a statement about a flow -- which is
    why it is stored in its own table rather than jammed into the per-flow
    predictions.
    """

    kind: str                 # "vertical" | "horizontal"
    src_ip: str
    distinct: int             # distinct ports (vertical) or hosts (horizontal)
    threshold: int
    window_seconds: float
    first_seen: float
    last_seen: float
    observations: int         # total events from this source still in the window
    sample: Tuple[str, ...]   # a few example ports or hosts

    def describe(self) -> str:
        what = "ports" if self.kind == "vertical" else "hosts"
        return (
            f"{self.kind} scan: {self.src_ip} touched {self.distinct} distinct "
            f"{what} in {self.window_seconds:.0f}s "
            f"(threshold {self.threshold}); e.g. {', '.join(self.sample)}"
        )


@dataclass
class _SourceWindow:
    """Sliding window of observations for one source address.

    ``events`` preserves arrival order so expiry is a pop from the left;
    ``ports`` and ``hosts`` are Counters so that "how many distinct" is
    ``len()`` -- O(1) -- rather than a scan of the deque.  Each event is
    pushed exactly once and popped exactly once, so maintenance is amortised
    O(1) per observation.

    A plain ``set`` of ports cannot work here: when an observation ages out
    you cannot tell whether its port should leave the set, because the set has
    forgotten how many times that port was seen.  That is the whole reason for
    carrying counts alongside the ordered log.
    """

    events: Deque[Tuple[float, int, str]] = field(default_factory=deque)
    ports: Counter = field(default_factory=Counter)
    hosts: Counter = field(default_factory=Counter)
    first_seen: float = 0.0
    last_seen: float = 0.0
    cooldown_until: Dict[str, float] = field(default_factory=dict)

    def add(self, now: float, dst_port: int, dst_ip: str) -> None:
        if not self.events:
            self.first_seen = now
        self.events.append((now, dst_port, dst_ip))
        self.ports[dst_port] += 1
        self.hosts[dst_ip] += 1
        self.last_seen = now

    def expire(self, cutoff: float) -> None:
        """Drop observations older than ``cutoff``. Amortised O(1) per event."""
        ev = self.events
        while ev and ev[0][0] < cutoff:
            _ts, port, host = ev.popleft()
            if self.ports[port] <= 1:
                del self.ports[port]
            else:
                self.ports[port] -= 1
            if self.hosts[host] <= 1:
                del self.hosts[host]
            else:
                self.hosts[host] -= 1
        self.first_seen = ev[0][0] if ev else 0.0

    @property
    def is_empty(self) -> bool:
        return not self.events


class ScanTracker:
    """Sliding-window scan detector, keyed by source address.

    Thread-safe and self-contained.  Feed it observations; it returns alerts.
    """

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        vertical_port_threshold: int = VERTICAL_PORT_THRESHOLD,
        horizontal_host_threshold: int = HORIZONTAL_HOST_THRESHOLD,
        max_sources: int = MAX_SOURCES,
        alert_cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
        ignore_sources: Optional[Iterable[str]] = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_sources < 1:
            raise ValueError("max_sources must be at least 1")

        self.window_seconds = float(window_seconds)
        self.vertical_port_threshold = int(vertical_port_threshold)
        self.horizontal_host_threshold = int(horizontal_host_threshold)
        self.max_sources = int(max_sources)
        self.alert_cooldown_seconds = float(alert_cooldown_seconds)
        self.ignore_sources = set(ignore_sources or ())

        # OrderedDict as an LRU: observing moves a source to the end, so the
        # least-recently-observed source is always at the front and eviction
        # is popitem(last=False) -- O(1), no scan of the table.
        self._sources: "OrderedDict[str, _SourceWindow]" = OrderedDict()
        self._lock = threading.Lock()

        # Operational counters, for /live/stats and for knowing whether the
        # cap is actually being hit in production.
        self.stats = {
            "observations": 0,
            "alerts_vertical": 0,
            "alerts_horizontal": 0,
            "sources_evicted": 0,
            "alerts_suppressed": 0,
        }

    # -----------------------------------------------------------------
    # INGEST
    # -----------------------------------------------------------------

    def observe(
        self,
        src_ip: str,
        dst_ip: str,
        dst_port: int,
        now: Optional[float] = None,
    ) -> List[ScanAlert]:
        """Record one (source touched host:port) event; return any alerts.

        Called once per NEW flow, from the capture thread. Returns a list
        because a single observation can cross both thresholds at once -- a
        source sweeping many ports across many hosts is doing both things.
        """
        if now is None:
            now = time.time()
        if src_ip in self.ignore_sources:
            return []

        cutoff = now - self.window_seconds
        alerts: List[ScanAlert] = []

        with self._lock:
            win = self._sources.get(src_ip)
            if win is None:
                win = _SourceWindow()
                self._sources[src_ip] = win
                self._evict_if_needed_locked()
            else:
                self._sources.move_to_end(src_ip)

            win.add(now, int(dst_port), dst_ip)
            win.expire(cutoff)
            self.stats["observations"] += 1

            for kind, distinct, threshold, counter in (
                ("vertical", len(win.ports), self.vertical_port_threshold, win.ports),
                ("horizontal", len(win.hosts), self.horizontal_host_threshold, win.hosts),
            ):
                if distinct < threshold:
                    continue
                # Suppression. Without it every packet after the threshold is
                # crossed emits another alert -- thousands per second during
                # the scan you are trying to report once.
                if now < win.cooldown_until.get(kind, 0.0):
                    self.stats["alerts_suppressed"] += 1
                    continue
                win.cooldown_until[kind] = now + self.alert_cooldown_seconds
                self.stats[f"alerts_{kind}"] += 1
                alerts.append(
                    ScanAlert(
                        kind=kind,
                        src_ip=src_ip,
                        distinct=distinct,
                        threshold=threshold,
                        window_seconds=self.window_seconds,
                        first_seen=win.first_seen,
                        last_seen=win.last_seen,
                        observations=len(win.events),
                        sample=tuple(
                            str(k) for k, _ in counter.most_common(SAMPLE_SIZE)
                        ),
                    )
                )

        return alerts

    # -----------------------------------------------------------------
    # MAINTENANCE
    # -----------------------------------------------------------------

    def _evict_if_needed_locked(self) -> None:
        """Enforce the source cap. Caller must hold the lock.

        Least-recently-observed first. The alternative -- refusing new sources
        once full -- would let an attacker fill the table with decoys and then
        scan freely from an address the tracker has stopped accepting. Evicting
        the stalest entry keeps the table responsive to current traffic.
        """
        while len(self._sources) > self.max_sources:
            self._sources.popitem(last=False)
            self.stats["sources_evicted"] += 1

    def sweep(self, now: Optional[float] = None) -> int:
        """Drop expired observations and empty sources. Returns sources removed.

        Not required for correctness -- ``observe`` expires the source it
        touches -- but a source that stops transmitting would otherwise hold
        its window forever. Call periodically from the cleanup worker.
        """
        if now is None:
            now = time.time()
        cutoff = now - self.window_seconds
        removed = 0
        with self._lock:
            for src in list(self._sources.keys()):
                win = self._sources[src]
                win.expire(cutoff)
                if win.is_empty:
                    del self._sources[src]
                    removed += 1
        return removed

    # -----------------------------------------------------------------
    # READ
    # -----------------------------------------------------------------

    def snapshot(self, now: Optional[float] = None) -> Dict[str, object]:
        """Current state, for /live/stats. Takes the lock; never blocks on I/O."""
        if now is None:
            now = time.time()
        with self._lock:
            return {
                "tracked_sources": len(self._sources),
                "max_sources": self.max_sources,
                "window_seconds": self.window_seconds,
                "vertical_port_threshold": self.vertical_port_threshold,
                "horizontal_host_threshold": self.horizontal_host_threshold,
                **dict(self.stats),
            }

    def top_sources(self, limit: int = 10, now: Optional[float] = None) -> List[dict]:
        """The busiest sources by distinct-port count, for the dashboard."""
        if now is None:
            now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            rows = []
            for src, win in self._sources.items():
                win.expire(cutoff)
                if win.is_empty:
                    continue
                rows.append({
                    "src_ip": src,
                    "distinct_ports": len(win.ports),
                    "distinct_hosts": len(win.hosts),
                    "observations": len(win.events),
                    "last_seen": win.last_seen,
                })
        rows.sort(key=lambda r: (-r["distinct_ports"], -r["distinct_hosts"]))
        return rows[:limit]

    def reset(self) -> None:
        with self._lock:
            self._sources.clear()
            for k in self.stats:
                self.stats[k] = 0


# Module-level instance used by the capture path. In-memory and ephemeral by
# design: this is operational state, not a record. If the process restarts the
# window rebuilds within ``WINDOW_SECONDS``, and persisting it would buy a
# minute of history at the cost of a durability story nobody needs.
scan_tracker = ScanTracker()
