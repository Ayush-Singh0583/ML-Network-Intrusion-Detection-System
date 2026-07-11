from backend.live.statistics import (
    safe_mean,
    safe_std,
    safe_max,
    safe_min,
    safe_var,
    calculate_iat,
    iat_total,
    iat_mean,
    iat_std,
    iat_max,
    iat_min,
    active_mean,
    active_std,
    active_max,
    active_min,
    idle_mean,
    idle_std,
    idle_max,
    idle_min,
    bulk_rate,
    packets_per_bulk,
    bytes_per_bulk
)


def extract_features(flow):

    duration = max(flow.last_seen - flow.first_seen, 0.000001)

    packet_lengths = flow.packet_lengths
    fwd_lengths = flow.forward_lengths
    bwd_lengths = flow.backward_lengths

    flow_iat = calculate_iat(flow.timestamps)
    fwd_iat = calculate_iat(flow.forward_timestamps)
    bwd_iat = calculate_iat(flow.backward_timestamps)

    features = {

        ##################################################
        # BASIC
        ##################################################

        "Destination Port": flow.dst_port,
        "Flow Duration": duration,

        "Total Fwd Packets": flow.forward_packets,
        "Total Backward Packets": flow.backward_packets,

        ##################################################
        # LENGTHS
        ##################################################

        "Total Length of Fwd Packets": flow.forward_bytes,
        "Total Length of Bwd Packets": flow.backward_bytes,

        "Fwd Packet Length Max": safe_max(fwd_lengths),
        "Fwd Packet Length Min": safe_min(fwd_lengths),
        "Fwd Packet Length Mean": safe_mean(fwd_lengths),
        "Fwd Packet Length Std": safe_std(fwd_lengths),

        "Bwd Packet Length Max": safe_max(bwd_lengths),
        "Bwd Packet Length Min": safe_min(bwd_lengths),
        "Bwd Packet Length Mean": safe_mean(bwd_lengths),
        "Bwd Packet Length Std": safe_std(bwd_lengths),

        ##################################################
        # FLOW RATE
        ##################################################

        "Flow Bytes/s": flow.bytes / duration,
        "Flow Packets/s": flow.packets / duration,

        ##################################################
        # FLOW IAT
        ##################################################

        "Flow IAT Mean": iat_mean(flow_iat),
        "Flow IAT Std": iat_std(flow_iat),
        "Flow IAT Max": iat_max(flow_iat),
        "Flow IAT Min": iat_min(flow_iat),

        ##################################################
        # FWD IAT
        ##################################################

        "Fwd IAT Total": iat_total(fwd_iat),
        "Fwd IAT Mean": iat_mean(fwd_iat),
        "Fwd IAT Std": iat_std(fwd_iat),
        "Fwd IAT Max": iat_max(fwd_iat),
        "Fwd IAT Min": iat_min(fwd_iat),

        ##################################################
        # BWD IAT
        ##################################################

        "Bwd IAT Total": iat_total(bwd_iat),
        "Bwd IAT Mean": iat_mean(bwd_iat),
        "Bwd IAT Std": iat_std(bwd_iat),
        "Bwd IAT Max": iat_max(bwd_iat),
        "Bwd IAT Min": iat_min(bwd_iat),

        ##################################################
        # FLAGS
        ##################################################

        "Fwd PSH Flags": flow.psh_count,
        "Bwd PSH Flags": 0,

        "Fwd URG Flags": flow.urg_count,
        "Bwd URG Flags": 0,

        ##################################################
        # HEADER
        ##################################################

        "Fwd Header Length": flow.forward_header_length,
        "Bwd Header Length": flow.backward_header_length,

        ##################################################
        # RATE
        ##################################################

        "Fwd Packets/s": flow.forward_packets / duration,
        "Bwd Packets/s": flow.backward_packets / duration,

        ##################################################
        # PACKET STATS
        ##################################################

        "Min Packet Length": safe_min(packet_lengths),
        "Max Packet Length": safe_max(packet_lengths),
        "Packet Length Mean": safe_mean(packet_lengths),
        "Packet Length Std": safe_std(packet_lengths),
        "Packet Length Variance": safe_var(packet_lengths),

        ##################################################
        # TCP FLAGS
        ##################################################

        "FIN Flag Count": flow.fin_count,
        "SYN Flag Count": flow.syn_count,
        "RST Flag Count": flow.rst_count,
        "PSH Flag Count": flow.psh_count,
        "ACK Flag Count": flow.ack_count,
        "URG Flag Count": flow.urg_count,
        "CWE Flag Count": flow.cwe_count,
        "ECE Flag Count": flow.ece_count,

        ##################################################
        # RATIO
        ##################################################

        "Down/Up Ratio": flow.backward_packets / max(flow.forward_packets, 1),

        ##################################################
        # AVERAGES
        ##################################################

        "Average Packet Size": safe_mean(packet_lengths),
        "Avg Fwd Segment Size": safe_mean(fwd_lengths),
        "Avg Bwd Segment Size": safe_mean(bwd_lengths),

        "Fwd Header Length.1": flow.forward_header_length,

        ##################################################
        # BULK
        ##################################################

        "Fwd Avg Bytes/Bulk": bytes_per_bulk(flow.fwd_bulk_bytes),
        "Fwd Avg Packets/Bulk": packets_per_bulk(flow.fwd_bulk_packets),
        "Fwd Avg Bulk Rate": bulk_rate(flow.fwd_bulk_bytes, duration),

        "Bwd Avg Bytes/Bulk": bytes_per_bulk(flow.bwd_bulk_bytes),
        "Bwd Avg Packets/Bulk": packets_per_bulk(flow.bwd_bulk_packets),
        "Bwd Avg Bulk Rate": bulk_rate(flow.bwd_bulk_bytes, duration),

        ##################################################
        # SUBFLOW
        ##################################################

        "Subflow Fwd Packets": flow.forward_packets,
        "Subflow Fwd Bytes": flow.forward_bytes,

        "Subflow Bwd Packets": flow.backward_packets,
        "Subflow Bwd Bytes": flow.backward_bytes,

        ##################################################
        # WINDOW
        ##################################################

        "Init_Win_bytes_forward": flow.init_win_bytes_forward,
        "Init_Win_bytes_backward": flow.init_win_bytes_backward,

        ##################################################
        # SEGMENT
        ##################################################

        "act_data_pkt_fwd": flow.act_data_pkt_fwd,
        "min_seg_size_forward": flow.min_seg_size_forward,

        ##################################################
        # ACTIVE
        ##################################################

        "Active Mean": active_mean(flow),
        "Active Std": active_std(flow),
        "Active Max": active_max(flow),
        "Active Min": active_min(flow),

        ##################################################
        # IDLE
        ##################################################

        "Idle Mean": idle_mean(flow),
        "Idle Std": idle_std(flow),
        "Idle Max": idle_max(flow),
        "Idle Min": idle_min(flow),
    }

    return features