import time
import threading

from scapy.layers.inet import IP, TCP, UDP

from backend.live.models import Flow

flows = {}
flows_lock = threading.Lock()

# ==========================================
# LIVE COUNTERS
# ==========================================

live_stats = {

    "packets": 0,

    "bytes": 0,

    "flows_created": 0,

    "flows_finished": 0,

    "start_time": time.time()

}
def create_flow_key(
    src_ip,
    dst_ip,
    src_port,
    dst_port,
    protocol
):

    left = (src_ip, src_port)
    right = (dst_ip, dst_port)

    if left <= right:
        return (
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            protocol
        )

    return (
        dst_ip,
        dst_port,
        src_ip,
        src_port,
        protocol
    )

def process_packet(packet):

    if not packet.haslayer(IP):
        return

    ip = packet[IP]

    src_ip = ip.src
    dst_ip = ip.dst
    protocol = ip.proto

    src_port = 0
    dst_port = 0

    tcp = None

    if packet.haslayer(TCP):

        tcp = packet[TCP]

        src_port = tcp.sport
        dst_port = tcp.dport

    elif packet.haslayer(UDP):

        udp = packet[UDP]

        src_port = udp.sport
        dst_port = udp.dport

    flow_key = create_flow_key(
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        protocol
    )

    flow_id = str(flow_key)

    current_time = time.time()

    with flows_lock:

        # ----------------------------------
        # Create Flow
        # ----------------------------------

        if flow_id not in flows:

            flows[flow_id] = Flow(

                flow_id=flow_id,

                src_ip=flow_key[0],
                dst_ip=flow_key[2],

                src_port=flow_key[1],
                dst_port=flow_key[3],

                protocol=protocol

            )

            live_stats["flows_created"] += 1

        # Get Flow Object
        flow = flows[flow_id]

        # ----------------------------------
        # Time
        # ----------------------------------

        flow.last_seen = current_time

        if flow.last_packet_time is not None:

            gap = current_time - flow.last_packet_time

            if gap > 1:

                flow.idle_times.append(gap)

            else:

                flow.active_times.append(gap)

        flow.last_packet_time = current_time

        # ----------------------------------
        # Direction
        # ----------------------------------

        forward = (

            src_ip == flow.src_ip

            and

            src_port == flow.src_port

        )

        packet_length = len(packet)

        # ----------------------------------
        # Live Counters
        # ----------------------------------

        live_stats["packets"] += 1
        live_stats["bytes"] += packet_length

        # ----------------------------------
        # Flow Statistics
        # ----------------------------------

        flow.packets += 1
        flow.bytes += packet_length

        flow.packet_lengths.append(packet_length)
        flow.timestamps.append(current_time)

        # ----------------------------------
        # Forward / Backward
        # ----------------------------------

        if forward:

            flow.forward_packets += 1
            flow.forward_bytes += packet_length

            flow.forward_lengths.append(packet_length)
            flow.forward_timestamps.append(current_time)

            flow.fwd_bulk_packets += 1
            flow.fwd_bulk_bytes += packet_length

        else:

            flow.backward_packets += 1
            flow.backward_bytes += packet_length

            flow.backward_lengths.append(packet_length)
            flow.backward_timestamps.append(current_time)

            flow.bwd_bulk_packets += 1
            flow.bwd_bulk_bytes += packet_length

        # ----------------------------------
        # TCP Information
        # ----------------------------------

        if tcp:

            if forward:

                if flow.init_win_bytes_forward == 0:
                    flow.init_win_bytes_forward = tcp.window

            else:

                if flow.init_win_bytes_backward == 0:
                    flow.init_win_bytes_backward = tcp.window

            payload_length = len(tcp.payload)

            if forward and payload_length > 0:
                flow.act_data_pkt_fwd += 1

            if forward:

                if flow.min_seg_size_forward == 0:

                    flow.min_seg_size_forward = packet_length

                else:

                    flow.min_seg_size_forward = min(

                        flow.min_seg_size_forward,

                        packet_length

                    )

            flags = tcp.flags

            if flags.F:
                flow.fin_count += 1

            if flags.S:
                flow.syn_count += 1

            if flags.R:
                flow.rst_count += 1

            if flags.P:
                flow.psh_count += 1

            if flags.A:
                flow.ack_count += 1

            if flags.U:
                flow.urg_count += 1

            if flags.E:
                flow.ece_count += 1

            if hasattr(flags, "C") and flags.C:
                flow.cwe_count += 1

            header_length = tcp.dataofs * 4

        else:

            header_length = 8

        # ----------------------------------
        # Header Length
        # ----------------------------------

        if forward:

            flow.forward_header_length += header_length

        else:

            flow.backward_header_length += header_length

    print(

        f"[FLOW] "

        f"{flow.src_ip}:{flow.src_port}"

        f" <-> "

        f"{flow.dst_ip}:{flow.dst_port}"

        f" | Total={flow.packets}"

        f" | FWD={flow.forward_packets}"

        f" | BWD={flow.backward_packets}"

        f" | Bytes={flow.bytes}"

    )