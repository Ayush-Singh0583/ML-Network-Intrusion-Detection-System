import math


def safe_mean(values):

    if not values:
        return 0

    return sum(values) / len(values)


def safe_std(values):

    if len(values) < 2:
        return 0

    mean = safe_mean(values)

    variance = sum((x - mean) ** 2 for x in values) / len(values)

    return math.sqrt(variance)


def safe_var(values):

    if len(values) < 2:
        return 0

    mean = safe_mean(values)

    return sum((x - mean) ** 2 for x in values) / len(values)


def safe_max(values):

    if not values:
        return 0

    return max(values)


def safe_min(values):

    if not values:
        return 0

    return min(values)


# ======================================================
# Inter Arrival Time Statistics
# ======================================================

def calculate_iat(timestamps):

    if len(timestamps) < 2:
        return []

    return [

        timestamps[i] - timestamps[i - 1]

        for i in range(1, len(timestamps))

    ]


def iat_total(iats):

    return sum(iats)


def iat_mean(iats):

    return safe_mean(iats)


def iat_std(iats):

    return safe_std(iats)


def iat_max(iats):

    return safe_max(iats)


def iat_min(iats):

    return safe_min(iats)


# ======================================================
# Active / Idle Statistics
# ======================================================

def active_mean(flow):

    return safe_mean(flow.active_times)


def active_std(flow):

    return safe_std(flow.active_times)


def active_max(flow):

    return safe_max(flow.active_times)


def active_min(flow):

    return safe_min(flow.active_times)


def idle_mean(flow):

    return safe_mean(flow.idle_times)


def idle_std(flow):

    return safe_std(flow.idle_times)


def idle_max(flow):

    return safe_max(flow.idle_times)


def idle_min(flow):

    return safe_min(flow.idle_times)


# ======================================================
# Bulk Features
# ======================================================

def bulk_rate(bytes_count, duration):

    if duration <= 0:
        return 0

    return bytes_count / duration


def packets_per_bulk(packet_count):

    return packet_count


def bytes_per_bulk(byte_count):

    return byte_count