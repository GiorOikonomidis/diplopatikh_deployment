"""Shared constants for the deployment pipeline (fetch_models.py, run_onnx.py):
router/expert class orderings and the raw feature-column order the router
and every expert were trained on.
"""

EXPERT_NAMES: list[str] = ["G1", "G3", "G2"]

G1_CLASSES: list[str] = ["MITM-ArpSpoofing", "DNS_Spoofing", "Recon-HostDiscovery",
                        "DictionaryBruteForce", "BrowserHijacking"]
G3_CLASSES: list[str] = ["Recon-OSScan", "Recon-PortScan", "Recon-PingSweep", "SqlInjection"]
G2_CLASSES: list[str] = ["Uploading_Attack", "XSS", "Backdoor_Malware", "CommandInjection"]

EXPERT_CLASSES: dict[str, list[str]] = {"G1": G1_CLASSES, "G3": G3_CLASSES, "G2": G2_CLASSES}

# Raw input feature order (tile_knn_25_OVN_09), matching what the router's
# scaler and every LightGBM expert were fit/trained on.
FEATURE_NAMES: list[str] = [
    "Weight", "IAT", "Number", "Variance", "Duration", "Min",
    "Protocol Type", "Magnitue", "urg_count", "ack_count",
    "syn_count", "Tot size", "Header_Length", "Rate", "AVG",
    "Max", "rst_count", "SSH", "Tot sum", "Srate", "flow_duration",
    "Covariance", "Radius", "Std", "TCP",
]

# Router output order (index -> class name), fixed by the label_map the
# router was trained/scaled against (meta/label_map_tile_knn_25_OVN_09.joblib).
CLASSES: list[str] = [
    "DDoS-ACK_Fragmentation", "DDoS-SYN_Flood", "DDoS-PSHACK_Flood", "Mirai-greeth_flood",
    "DoS-SYN_Flood", "DDoS-TCP_Flood", "Mirai-greip_flood", "Mirai-udpplain",
    "DDoS-ICMP_Flood", "DDoS-SynonymousIP_Flood", "DDoS-UDP_Flood", "DDoS-UDP_Fragmentation",
    "DoS-UDP_Flood", "DDoS-ICMP_Fragmentation", "DDoS-RSTFINFlood", "DoS-TCP_Flood",
    "DDoS-HTTP_Flood", "DoS-HTTP_Flood", "DDoS-SlowLoris", "VulnerabilityScan",
    "BenignTraffic", "G1", "G3", "G2",
]
