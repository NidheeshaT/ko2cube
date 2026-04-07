"""
Carbon data providers for the Ko2cube environment.

This module provides a pluggable interface for carbon intensity and pricing data,
supporting multiple data sources including static CSV files and live carbon APIs.
"""

from server.providers.base import CarbonProvider, CarbonReading
from server.providers.config import REGION_MAPPINGS, get_provider_region
from server.providers.static import StaticCarbonProvider
from server.providers.watttime import WattTimeProvider
from server.providers.electricity_map import ElectricityMapProvider

__all__ = [
    "CarbonProvider",
    "CarbonReading",
    "StaticCarbonProvider",
    "WattTimeProvider",
    "ElectricityMapProvider",
    "REGION_MAPPINGS",
    "get_provider_region",
]
