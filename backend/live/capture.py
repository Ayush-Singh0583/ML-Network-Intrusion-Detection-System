from threading import Event

from scapy.all import sniff

from backend.live.flow_manager import process_packet


capture_running = Event()


def is_capture_running():

    return capture_running.is_set()


def start_capture(interface=None):

    print("Starting Live Packet Capture...")

    capture_running.set()

    sniff(
        iface=interface,
        prn=process_packet,
        store=False,
        stop_filter=lambda x: not capture_running.is_set()
    )

    print("Capture Stopped")


def stop_capture():

    capture_running.clear()