"""
Live packet capture, on scapy's AsyncSniffer so that Stop really stops.

The previous version ran a blocking ``sniff()`` whose ``stop_filter`` was only
checked when a packet arrived. Two bugs followed, both reproduced:

* Stop did not stop. The old thread lived until the next packet, and a Start
  in that window set the flag again, so the old thread kept going next to the
  new one. Every packet was then processed twice (measured: 2.00 calls per
  packet on the wire), which doubled packet counts and put zero-length gaps
  into the flow features.
* A failed start looked like a running capture. If the interface could not be
  opened (no admin rights, or Npcap missing on Windows) the thread died, but
  /live/status kept saying running=true.
"""

import threading

from scapy.all import AsyncSniffer

from backend.live.flow_manager import process_packet

START_TIMEOUT = 5.0     # seconds to wait for the interface to open

_lock = threading.Lock()
_sniffer = None


class CaptureError(RuntimeError):
    """The capture interface could not be opened."""


def _alive(sniffer):
    return (
        sniffer is not None
        and sniffer.thread is not None
        and sniffer.thread.is_alive()
    )


def is_capture_running():
    # Ask the thread, not a flag: the old flag stayed set after the thread died.
    return _alive(_sniffer)


def start_capture(interface=None):
    """Start sniffing. Returns False if a capture is already running.

    Raises CaptureError when the interface cannot be opened, instead of
    reporting a capture that is not happening.
    """
    global _sniffer

    with _lock:

        if _alive(_sniffer):
            return False

        opened = threading.Event()

        sniffer = AsyncSniffer(
            iface=interface,
            prn=process_packet,
            store=False,
            started_callback=opened.set,
        )
        sniffer.start()

        # Wait until the socket is open, or the thread has died trying.
        steps = int(START_TIMEOUT / 0.05)
        for _ in range(steps):
            if opened.wait(0.05) or not sniffer.thread.is_alive():
                break

        if not opened.is_set() and not sniffer.thread.is_alive():
            exc = sniffer.exception
            reason = f"{type(exc).__name__}: {exc}" if exc else "sniffer exited before it started"
            print(f"Live capture failed to start -- {reason}")
            raise CaptureError(reason)

        _sniffer = sniffer
        print("Starting Live Packet Capture...")
        return True


def stop_capture():
    """Stop sniffing and wait for the thread to exit.

    Returns False if nothing was running.
    """
    global _sniffer

    with _lock:

        sniffer, _sniffer = _sniffer, None

        if not _alive(sniffer):
            return False

        try:
            # Wakes the blocking select and joins the thread.
            sniffer.stop()
        except Exception:                             # noqa: BLE001
            # It may have died on its own in the meantime; either way it is
            # not running any more.
            pass

        print("Capture Stopped")
        return True
