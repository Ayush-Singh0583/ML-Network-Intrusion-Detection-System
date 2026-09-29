"""
Regression tests for three live-pipeline bugs, each reproduced before the fix.

1. Flow direction.  The Flow took its src/dst from the SORTED flow key, so
   "forward" meant "whichever IP sorts first as a string" rather than the
   first packet's direction (CICFlowMeter's definition, which the model was
   trained on). Fwd/Bwd features were swapped for many connections.
2. Stop did not stop.  A Stop then Start left two sniffers running and every
   packet was processed twice.
3. A failed start (no admin rights / Npcap missing) reported success, and
   /live/status said running=true with nothing capturing.

Plus the report footer, which printed one old run's accuracy for every run.

No root, no network, no trained model: packets are built in memory and the
sniffer is a fake with the same interface as scapy's AsyncSniffer.
"""

from __future__ import annotations

import pathlib
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scapy.layers.inet import IP, TCP  # noqa: E402

import backend.live.capture as capture  # noqa: E402
import backend.live.flow_manager as fm  # noqa: E402
from backend.live.extractor import extract_features  # noqa: E402
from backend.live.scan_tracker import scan_tracker  # noqa: E402
from backend.routers import live as live_router  # noqa: E402


# =====================================================================
# 1. FLOW DIRECTION
# =====================================================================


@pytest.fixture
def clean_flows(monkeypatch):
    monkeypatch.setattr(fm, "print", lambda *a, **k: None, raising=False)
    fm.flows.clear()
    yield fm.flows
    fm.flows.clear()
    scan_tracker.reset()


def _wire(pkt):
    """Round-trip through bytes so length/offset fields are filled in, as on a
    sniffed packet."""
    return IP(bytes(pkt))


def test_forward_is_the_first_packets_direction(clean_flows):
    # Client 192.168.1.10 opens a connection to server 10.0.0.1:443.
    # As strings "10.0.0.1" < "192.168.1.10", so the sorted key puts the
    # SERVER first -- the case the old code got backwards.
    client, server = "192.168.1.10", "10.0.0.1"
    fm.process_packet(_wire(IP(src=client, dst=server) / TCP(sport=54321, dport=443, flags="S", window=64240)))
    fm.process_packet(_wire(IP(src=server, dst=client) / TCP(sport=443, dport=54321, flags="SA", window=29200)))
    fm.process_packet(_wire(IP(src=client, dst=server) / TCP(sport=54321, dport=443, flags="A", window=64240)))

    (flow,) = clean_flows.values()
    assert (flow.src_ip, flow.src_port) == (client, 54321)
    assert (flow.dst_ip, flow.dst_port) == (server, 443)
    assert flow.forward_packets == 2 and flow.backward_packets == 1

    f = extract_features(flow)
    assert f["Destination Port"] == 443
    assert f["Total Fwd Packets"] == 2
    assert f["Init_Win_bytes_forward"] == 64240      # the client's window
    assert f["Init_Win_bytes_backward"] == 29200


def test_both_directions_still_share_one_flow(clean_flows):
    a, b = "10.0.0.1", "192.168.1.10"
    fm.process_packet(_wire(IP(src=a, dst=b) / TCP(sport=1111, dport=80)))
    fm.process_packet(_wire(IP(src=b, dst=a) / TCP(sport=80, dport=1111)))
    assert len(clean_flows) == 1


# =====================================================================
# 2 + 3. CAPTURE START / STOP / STATUS
# =====================================================================


class FakeSniffer:
    """Same surface as scapy.AsyncSniffer: start/stop/thread/exception."""

    instances: list = []
    fail_with: Exception | None = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.thread = None
        self.exception = None
        self.running = False
        self._stop = threading.Event()
        FakeSniffer.instances.append(self)

    def start(self):
        def run():
            self.running = True
            if FakeSniffer.fail_with is not None:
                self.exception = FakeSniffer.fail_with
                return                                  # dies before opening
            self.kwargs["started_callback"]()
            self._stop.wait()
            self.running = False

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def stop(self, join=True):
        self._stop.set()
        self.thread.join()


@pytest.fixture
def fake_sniffer(monkeypatch):
    FakeSniffer.instances = []
    FakeSniffer.fail_with = None
    monkeypatch.setattr(capture, "AsyncSniffer", FakeSniffer)
    monkeypatch.setattr(capture, "_sniffer", None)
    yield FakeSniffer
    capture.stop_capture()


def _alive(instances):
    return [s for s in instances if s.thread is not None and s.thread.is_alive()]


def test_stop_then_start_leaves_exactly_one_sniffer(fake_sniffer):
    assert live_router.start_live_capture()["message"] == "Live packet capture started."
    assert live_router.stop_live_capture()["message"] == "Live packet capture stopped."
    assert _alive(fake_sniffer.instances) == [], "stop returned while the sniffer still ran"

    live_router.start_live_capture()
    assert len(_alive(fake_sniffer.instances)) == 1
    assert live_router.capture_status() == {"running": True}


def test_second_start_does_not_spawn_a_second_sniffer(fake_sniffer):
    live_router.start_live_capture()
    assert live_router.start_live_capture()["message"] == "Live capture is already running."
    assert len(_alive(fake_sniffer.instances)) == 1


def test_failed_start_is_an_error_not_a_running_capture(fake_sniffer):
    from fastapi import HTTPException

    fake_sniffer.fail_with = PermissionError("Operation not permitted")
    with pytest.raises(HTTPException) as exc:
        live_router.start_live_capture()
    assert exc.value.status_code == 503
    assert "Operation not permitted" in exc.value.detail
    assert live_router.capture_status() == {"running": False}


def test_stop_when_idle_is_harmless(fake_sniffer):
    assert live_router.stop_live_capture()["message"] == "Capture is not running."


# =====================================================================
# REPORT FOOTER STATES THIS RUN'S ACCURACY
# =====================================================================


def test_detection_footer_uses_this_runs_accuracy():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    from evaluation import format_detection_table

    df = pd.DataFrame([
        {"fpr_budget": 0.01, "threshold": 0.5, "class": "DDoS", "n": 10,
         "detection_rate": 0.9, "observed_benign_fpr": 0.01, "false_alerts_per_day": 1.0},
        {"fpr_budget": 0.01, "threshold": 0.5, "class": "__ANY_ATTACK__", "n": 10,
         "detection_rate": 0.9, "observed_benign_fpr": 0.01, "false_alerts_per_day": 1.0,
         "projected_true_alerts_per_day": 9.0, "projected_precision": 0.9},
    ])
    text = " ".join(format_detection_table(
        df, accuracy={"multi_class": 0.93, "binary": 0.97, "all_benign_baseline": 0.6}).split())
    assert "93.00% multi-class" in text and "97.00% binary" in text
    assert "58.47%" not in text, "footer still prints the old hardcoded run"
    assert "below the constant-function baseline" not in text   # 93% > 60%

    assert "58.47%" not in format_detection_table(df)           # no numbers at all
