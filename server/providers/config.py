"""
Region mapping configuration for carbon data providers.

Maps cloud provider regions (AWS, GCP, Azure) to grid operator regions
used by carbon intensity APIs (WattTime, Electricity Map).
"""

from typing import Dict, Optional

# Cloud region -> Carbon API region mappings
# Each cloud region maps to provider-specific region identifiers
REGION_MAPPINGS: Dict[str, Dict[str, str]] = {
    # AWS Regions
    "us-east-1": {
        "watttime": "PJM_WEST",
        "electricity_map": "US-MIDA-PJM",
        "grid": "PJM Interconnection",
    },
    "us-east-2": {
        "watttime": "PJM_WEST",
        "electricity_map": "US-MIDA-PJM",
        "grid": "PJM Interconnection",
    },
    "us-west-1": {
        "watttime": "CAISO_NP15",
        "electricity_map": "US-CAL-CISO",
        "grid": "California ISO",
    },
    "us-west-2": {
        "watttime": "CAISO_NP15",
        "electricity_map": "US-CAL-CISO",
        "grid": "California ISO",
    },
    "eu-west-1": {
        "watttime": "IE",
        "electricity_map": "IE",
        "grid": "Ireland Grid",
    },
    "eu-west-2": {
        "watttime": "UK",
        "electricity_map": "GB",
        "grid": "UK National Grid",
    },
    "eu-central-1": {
        "watttime": "DE",
        "electricity_map": "DE",
        "grid": "Germany Grid",
    },
    "eu-north-1": {
        "watttime": "SE",
        "electricity_map": "SE",
        "grid": "Sweden Grid",
    },
    "ap-northeast-1": {
        "watttime": "JP_TEPCO",
        "electricity_map": "JP-TK",
        "grid": "Tokyo Electric",
    },
    "ap-southeast-1": {
        "watttime": "SG",
        "electricity_map": "SG",
        "grid": "Singapore Grid",
    },
    "ap-south-1": {
        "watttime": "IN_WR",
        "electricity_map": "IN-WE",
        "grid": "Western India Grid",
    },
    
    # GCP Regions
    "us-central1": {
        "watttime": "MISO_MI",
        "electricity_map": "US-MIDW-MISO",
        "grid": "MISO",
    },
    "us-east4": {
        "watttime": "PJM_WEST",
        "electricity_map": "US-MIDA-PJM",
        "grid": "PJM Interconnection",
    },
    "europe-west1": {
        "watttime": "BE",
        "electricity_map": "BE",
        "grid": "Belgium Grid",
    },
    "europe-west4": {
        "watttime": "NL",
        "electricity_map": "NL",
        "grid": "Netherlands Grid",
    },
    "europe-north1": {
        "watttime": "FI",
        "electricity_map": "FI",
        "grid": "Finland Grid",
    },
    "asia-east1": {
        "watttime": "TW",
        "electricity_map": "TW",
        "grid": "Taiwan Grid",
    },
    
    # Azure Regions
    "eastus": {
        "watttime": "PJM_WEST",
        "electricity_map": "US-MIDA-PJM",
        "grid": "PJM Interconnection",
    },
    "westus2": {
        "watttime": "CAISO_NP15",
        "electricity_map": "US-CAL-CISO",
        "grid": "California ISO",
    },
    "northeurope": {
        "watttime": "IE",
        "electricity_map": "IE",
        "grid": "Ireland Grid",
    },
    "westeurope": {
        "watttime": "NL",
        "electricity_map": "NL",
        "grid": "Netherlands Grid",
    },
    "uksouth": {
        "watttime": "UK",
        "electricity_map": "GB",
        "grid": "UK National Grid",
    },
}

# Typical carbon intensity ranges by region (gCO2/kWh)
# Used for generating realistic synthetic data
REGION_CARBON_PROFILES: Dict[str, Dict[str, float]] = {
    "us-east-1": {"min": 200, "max": 500, "avg": 350},
    "us-west-2": {"min": 100, "max": 400, "avg": 200},
    "eu-west-1": {"min": 150, "max": 450, "avg": 300},
    "eu-north-1": {"min": 20, "max": 100, "avg": 50},
    "ap-northeast-1": {"min": 300, "max": 600, "avg": 450},
}

# Default fallback values
DEFAULT_CARBON_INTENSITY = 400.0  # gCO2/kWh
DEFAULT_SPOT_MULTIPLIER = 0.5


def get_provider_region(
    cloud_region: str,
    provider: str = "watttime"
) -> Optional[str]:
    """
    Get the carbon API region identifier for a cloud region.
    
    Args:
        cloud_region: Cloud provider region (e.g., 'us-east-1')
        provider: Carbon API provider ('watttime' or 'electricity_map')
        
    Returns:
        Provider-specific region identifier, or None if not mapped
    """
    mapping = REGION_MAPPINGS.get(cloud_region, {})
    return mapping.get(provider)


def get_all_mapped_regions(provider: str = "watttime") -> Dict[str, str]:
    """
    Get all cloud regions mapped to a specific provider.
    
    Args:
        provider: Carbon API provider name
        
    Returns:
        Dict of cloud_region -> provider_region
    """
    return {
        cloud: mapping[provider]
        for cloud, mapping in REGION_MAPPINGS.items()
        if provider in mapping
    }


def is_region_supported(cloud_region: str) -> bool:
    """
    Check if a cloud region has carbon API mappings.
    
    Args:
        cloud_region: Cloud provider region identifier
        
    Returns:
        True if region has mappings, False otherwise
    """
    return cloud_region in REGION_MAPPINGS


def get_region_profile(cloud_region: str) -> Dict[str, float]:
    """
    Get typical carbon intensity profile for a region.
    
    Args:
        cloud_region: Cloud provider region identifier
        
    Returns:
        Dict with 'min', 'max', 'avg' carbon intensity values
    """
    return REGION_CARBON_PROFILES.get(
        cloud_region,
        {"min": 200, "max": 500, "avg": DEFAULT_CARBON_INTENSITY}
    )
