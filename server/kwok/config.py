from kubernetes import config as k8s_config

# Cluster configs as Python dicts — no YAML files needed.
# Keys map directly to kwokctl flags (see cluster.py for translation).
CLUSTER_CONFIG: dict[str, dict] = {
    "default": {
        "region":             "us-east",
        "kubeApiserverPort":  34080,
        "labels": {"environment": "simulation", "default": "true"},
    },
    "us-east": {
        "region":             "us-east",
        "kubeApiserverPort":  34080,
        "labels": {"environment": "simulation"},
    },
    "us-west": {
        "region":             "us-west",
        "kubeApiserverPort":  34081,
        "labels": {"environment": "simulation"},
    },
    "eu-west": {
        "region":             "eu-west",
        "kubeApiserverPort":  34082,
        "labels": {"environment": "simulation"},
    },
    "test": {
        "region":             "test",
        "kubeApiserverPort":  34099,
        "labels": {"environment": "testing"},
    },
}


def get_cluster_config(region: str = "default") -> dict:
    """Return the cluster config dict for the given region."""
    if region not in CLUSTER_CONFIG:
        raise ValueError(f"Unknown region '{region}'. Valid options: {list(CLUSTER_CONFIG.keys())}")
    return CLUSTER_CONFIG[region]


def load_kwok_kubeconfig(cluster_name: str = "kwok-cluster") -> dict:
    """
    Loads the kubernetes client configuration from a dict built at runtime
    by querying 'kwokctl get kubeconfig'. No kubeconfig file needed.

    Args:
        cluster_name: The kwok cluster name (default: 'kwok-cluster')

    Returns:
        The kubeconfig dict that was loaded.
    """
    import subprocess
    import yaml
    from kubernetes import client as k8s_client

    result = subprocess.run(
        ["kwokctl", "get", "kubeconfig", "--name", cluster_name],
        capture_output=True,
        text=True,
        check=True,
    )
    kubeconfig_dict = yaml.safe_load(result.stdout)

    k8s_config.load_kube_config_from_dict(kubeconfig_dict)

    # Kwok uses a self-signed cert — disable SSL verification for local simulation
    k8s_client.configuration.verify_ssl = False

    print(f"Loaded kubeconfig for cluster '{cluster_name}' from dict.")
    return kubeconfig_dict
