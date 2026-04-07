"""
Abstract base class for carbon data providers.

Defines the interface that all carbon data providers must implement,
enabling the environment to work with different data sources
(static CSV, WattTime, Electricity Map, etc.) interchangeably.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CarbonReading:
    """A single carbon intensity reading for a region."""
    region: str
    intensity_gco2_kwh: float
    timestamp: str
    forecast: Optional[List[float]] = None
    spot_multiplier: float = 1.0


class CarbonProvider(ABC):
    """
    Abstract base class for carbon intensity data providers.
    
    Implementations must provide methods to:
    - Get current carbon intensity for a region
    - Get carbon intensity forecasts
    - Get spot price multipliers
    - List supported regions
    """

    @abstractmethod
    def get_current_intensity(self, region: str, step: int) -> float:
        """
        Get current carbon intensity for a region at given simulation step.
        
        Args:
            region: Cloud provider region identifier (e.g., 'us-east-1')
            step: Current simulation step
            
        Returns:
            Carbon intensity in gCO2/kWh
        """
        pass

    @abstractmethod
    def get_forecast(self, region: str, step: int, lookahead: int) -> List[float]:
        """
        Get carbon intensity forecast for upcoming steps.
        
        Args:
            region: Cloud provider region identifier
            step: Current simulation step
            lookahead: Number of steps to forecast
            
        Returns:
            List of predicted carbon intensities for next `lookahead` steps
        """
        pass

    @abstractmethod
    def get_spot_multiplier(self, region: str, step: int) -> float:
        """
        Get spot price multiplier for a region.
        
        The multiplier is applied to base on-demand prices to get spot prices.
        Values < 1.0 indicate spot discount, > 1.0 indicates spot premium.
        
        Args:
            region: Cloud provider region identifier
            step: Current simulation step
            
        Returns:
            Spot price multiplier (typically 0.3-0.7 for normal conditions)
        """
        pass

    @abstractmethod
    def get_supported_regions(self) -> List[str]:
        """
        Return list of supported region identifiers.
        
        Returns:
            List of region strings this provider can serve data for
        """
        pass

    def get_reading(self, region: str, step: int, lookahead: int = 6) -> CarbonReading:
        """
        Get a complete carbon reading including current intensity and forecast.
        
        Args:
            region: Cloud provider region identifier
            step: Current simulation step
            lookahead: Number of steps to forecast
            
        Returns:
            CarbonReading with intensity, forecast, and spot multiplier
        """
        return CarbonReading(
            region=region,
            intensity_gco2_kwh=self.get_current_intensity(region, step),
            timestamp=f"step_{step}",
            forecast=self.get_forecast(region, step, lookahead),
            spot_multiplier=self.get_spot_multiplier(region, step),
        )

    def get_all_readings(self, regions: List[str], step: int, lookahead: int = 6) -> Dict[str, CarbonReading]:
        """
        Get carbon readings for multiple regions.
        
        Args:
            regions: List of region identifiers
            step: Current simulation step
            lookahead: Number of steps to forecast
            
        Returns:
            Dict mapping region names to CarbonReading objects
        """
        return {r: self.get_reading(r, step, lookahead) for r in regions}

    def validate_region(self, region: str) -> bool:
        """
        Check if a region is supported by this provider.
        
        Args:
            region: Region identifier to check
            
        Returns:
            True if region is supported, False otherwise
        """
        return region in self.get_supported_regions()
