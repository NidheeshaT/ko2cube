"""
constants.py — AWS EC2 instance type specs for EKS node simulation.

These dicts mirror the real capacity and allocatable resources that
the corresponding EC2 instance type would report as a Kubernetes node.

Usage:
    from server.kwok.constants import EC2_INSTANCE_TYPES

    spec = EC2_INSTANCE_TYPES["m5.large"]
    print(spec["cpu"])       # "2"
    print(spec["memory"])    # "8Gi"
"""

# ---------------------------------------------------------------------------
# EC2 instance type → Kubernetes node capacity
# Format:
#   cpu      : vCPU count as string (Kubernetes resource quantity)
#   memory   : RAM as string (Kubernetes resource quantity)
#   pods     : max pods supported by the instance type on EKS
#   arch     : processor architecture
# ---------------------------------------------------------------------------

EC2_INSTANCE_TYPES: dict[str, dict] = {
    # --- General Purpose ---
    "t3.small":    {"cpu": "2",   "memory": "2Gi",   "pods": 11,  "arch": "amd64"},
    "t3.medium":   {"cpu": "2",   "memory": "4Gi",   "pods": 17,  "arch": "amd64"},
    "t3.large":    {"cpu": "2",   "memory": "8Gi",   "pods": 35,  "arch": "amd64"},
    "t3.xlarge":   {"cpu": "4",   "memory": "16Gi",  "pods": 58,  "arch": "amd64"},
    "t3.2xlarge":  {"cpu": "8",   "memory": "32Gi",  "pods": 58,  "arch": "amd64"},

    "m5.large":    {"cpu": "2",   "memory": "8Gi",   "pods": 29,  "arch": "amd64"},
    "m5.xlarge":   {"cpu": "4",   "memory": "16Gi",  "pods": 58,  "arch": "amd64"},
    "m5.2xlarge":  {"cpu": "8",   "memory": "32Gi",  "pods": 58,  "arch": "amd64"},
    "m5.4xlarge":  {"cpu": "16",  "memory": "64Gi",  "pods": 234, "arch": "amd64"},
    "m5.8xlarge":  {"cpu": "32",  "memory": "128Gi", "pods": 234, "arch": "amd64"},

    # --- Compute Optimised ---
    "c5.large":    {"cpu": "2",   "memory": "4Gi",   "pods": 29,  "arch": "amd64"},
    "c5.xlarge":   {"cpu": "4",   "memory": "8Gi",   "pods": 58,  "arch": "amd64"},
    "c5.2xlarge":  {"cpu": "8",   "memory": "16Gi",  "pods": 58,  "arch": "amd64"},
    "c5.4xlarge":  {"cpu": "16",  "memory": "32Gi",  "pods": 234, "arch": "amd64"},
    "c5.9xlarge":  {"cpu": "36",  "memory": "72Gi",  "pods": 234, "arch": "amd64"},

    # --- Memory Optimised ---
    "r5.large":    {"cpu": "2",   "memory": "16Gi",  "pods": 29,  "arch": "amd64"},
    "r5.xlarge":   {"cpu": "4",   "memory": "32Gi",  "pods": 58,  "arch": "amd64"},
    "r5.2xlarge":  {"cpu": "8",   "memory": "64Gi",  "pods": 58,  "arch": "amd64"},
    "r5.4xlarge":  {"cpu": "16",  "memory": "128Gi", "pods": 234, "arch": "amd64"},

    # --- Arm / Graviton ---
    "m6g.medium":  {"cpu": "1",   "memory": "4Gi",   "pods": 8,   "arch": "arm64"},
    "m6g.large":   {"cpu": "2",   "memory": "8Gi",   "pods": 29,  "arch": "arm64"},
    "m6g.xlarge":  {"cpu": "4",   "memory": "16Gi",  "pods": 58,  "arch": "arm64"},
    "m6g.2xlarge": {"cpu": "8",   "memory": "32Gi",  "pods": 58,  "arch": "arm64"},

    # --- GPU ---
    "p3.2xlarge":  {"cpu": "8",   "memory": "61Gi",  "pods": 58,  "arch": "amd64"},
    "p3.8xlarge":  {"cpu": "32",  "memory": "244Gi", "pods": 234, "arch": "amd64"},
    "g4dn.xlarge": {"cpu": "4",   "memory": "16Gi",  "pods": 58,  "arch": "amd64"},
    "g4dn.2xlarge":{"cpu": "8",   "memory": "32Gi",  "pods": 58,  "arch": "amd64"},
}
