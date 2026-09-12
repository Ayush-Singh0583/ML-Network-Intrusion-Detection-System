from dataclasses import dataclass, field
import time


@dataclass
class Flow:

    flow_id: str

    src_ip: str
    dst_ip: str

    src_port: int
    dst_port: int

    protocol: int

    # ==========================
    # Time
    # ==========================

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # ==========================
    # Overall Statistics
    # ==========================

    packets: int = 0
    bytes: int = 0

    forward_packets: int = 0
    backward_packets: int = 0

    forward_bytes: int = 0
    backward_bytes: int = 0

    # ==========================
    # Packet Lengths
    # ==========================

    packet_lengths: list = field(default_factory=list)

    forward_lengths: list = field(default_factory=list)
    backward_lengths: list = field(default_factory=list)

    # ==========================
    # Timestamps
    # ==========================

    timestamps: list = field(default_factory=list)

    forward_timestamps: list = field(default_factory=list)
    backward_timestamps: list = field(default_factory=list)

    # ==========================
    # Active / Idle Tracking
    # ==========================

    last_packet_time: float | None = None

    active_times: list = field(default_factory=list)
    idle_times: list = field(default_factory=list)

    # ==========================
    # TCP Flags
    # ==========================

    fin_count: int = 0
    syn_count: int = 0
    rst_count: int = 0
    psh_count: int = 0
    ack_count: int = 0
    urg_count: int = 0
    ece_count: int = 0
    cwe_count: int = 0

    # ==========================
    # Header Lengths
    # ==========================

    forward_header_length: int = 0
    backward_header_length: int = 0

    # ==========================
    # TCP Window
    # ==========================

    init_win_bytes_forward: int = 0
    init_win_bytes_backward: int = 0

    # ==========================
    # Segment Information
    # ==========================

    act_data_pkt_fwd: int = 0
    min_seg_size_forward: int = 0

    # ==========================
    # Bulk Statistics
    # ==========================

    fwd_bulk_bytes: int = 0
    fwd_bulk_packets: int = 0

    bwd_bulk_bytes: int = 0
    bwd_bulk_packets: int = 0