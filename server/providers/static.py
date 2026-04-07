"""
Static carbon data provider using CSV timeseries data.

This provider uses pre-generated CSV data for carbon intensities and spot prices,
making it suitable for deterministic training and testing scenarios.
"""

import csv
import os
from typing import Dict, List, Optional

from server.providers.base import CarbonProvider, CarbonReading


class StaticCarbonProvider(CarbonProvider):
    """
    Carbon provider using static timeseries data from CSV files.
    
    This is the default provider for training, ensuring reproducible
    carbon intensity and pricing data across episodes.
    
    The CSV file format expects columns:
    - carbon_{region}: Carbon intensity in gCO2/kWh
    - spot_mult_{region}: Spot price multiplier
    
    Example: carbon_us-east-1, spot_mult_us-east-1
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        regions: Optional[List[str]] = None,
    ):
        """
        Initialize the static carbon provider.
        
        Args:
            csv_path: Path to CSV file with timeseries data.
                     If None, uses default data/cleaned_timeseries_data.csv
            regions: List of regions to support.
                    If None, auto-detects from CSV columns
        """
        if csv_path is None:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(root_dir, "data", "cleaned_timeseries_data.csv")
        
        self._csv_path = csv_path
        self._timeseries: List[Dict[str, str]] = []
        self._regions: List[str] = regions or []
        
        self._load_data()

    def _load_data(self) -> None:
        """Load timeseries data from CSV file."""
        with open(self._csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._timeseries.append(row)
        
        if not self._regions and self._timeseries:
            self._regions = self._detect_regions()

    def _detect_regions(self) -> List[str]:
        """Auto-detect regions from CSV column names."""
        if not self._timeseries:
            return []
        
        regions = set()
        for col in self._timeseries[0].keys():
            if col.startswith("carbon_"):
                region = col.replace("carbon_", "")
                regions.add(region)
        
        return sorted(regions)

    def get_current_intensity(self, region: str, step: int) -> float:
        """Get current carbon intensity for a region at given step."""
        if not self._timeseries:
            return 0.0
        
        idx = step % len(self._timeseries)
        row = self._timeseries[idx]
        
        col_name = f"carbon_{region}"
        value = row.get(col_name, "0.0")
        return float(value)

    def get_forecast(self, region: str, step: int, lookahead: int) -> List[float]:
        """Get carbon intensity forecast for upcoming steps."""
        if not self._timeseries:
            return [0.0] * lookahead
        
        forecast = []
        for i in range(lookahead):
            idx = (step + i) % len(self._timeseries)
            row = self._timeseries[idx]
            col_name = f"carbon_{region}"
            value = float(row.get(col_name, "0.0"))
            forecast.append(value)
        
        return forecast

    def get_spot_multiplier(self, region: str, step: int) -> float:
        """Get spot price multiplier for a region."""
        if not self._timeseries:
            return 1.0
        
        idx = step % len(self._timeseries)
        row = self._timeseries[idx]
        
        col_name = f"spot_mult_{region}"
        value = row.get(col_name, "1.0")
        return float(value)

    def get_supported_regions(self) -> List[str]:
        """Return list of supported region identifiers."""
        return list(self._regions)

    @property
    def num_steps(self) -> int:
        """Return the number of steps in the timeseries data."""
        return len(self._timeseries)

    def get_row(self, step: int) -> Dict[str, str]:
        """
        Get the raw CSV row for a given step.
        
        Useful for accessing additional columns not part of the standard interface.
        
        Args:
            step: Simulation step
            
        Returns:
            Dict of column_name -> value for that row
        """
        if not self._timeseries:
            return {}
        idx = step % len(self._timeseries)
        return self._timeseries[idx]

    def get_all_intensities_at_step(self, step: int) -> Dict[str, float]:
        """
        Get carbon intensities for all regions at a given step.
        
        Args:
            step: Simulation step
            
        Returns:
            Dict of region -> carbon intensity
        """
        return {
            region: self.get_current_intensity(region, step)
            for region in self._regions
        }

    def get_min_intensity_region(self, step: int) -> str:
        """
        Find the region with lowest carbon intensity at a given step.
        
        Args:
            step: Simulation step
            
        Returns:
            Region name with minimum carbon intensity
        """
        intensities = self.get_all_intensities_at_step(step)
        if not intensities:
            return ""
        return min(intensities.keys(), key=lambda r: intensities[r])

    def get_average_intensity(self, region: str, start_step: int, end_step: int) -> float:
        """
        Calculate average carbon intensity over a range of steps.
        
        Args:
            region: Region identifier
            start_step: Start of range (inclusive)
            end_step: End of range (inclusive)
            
        Returns:
            Average carbon intensity over the range
        """
        if start_step > end_step or not self._timeseries:
            return 0.0
        
        total = 0.0
        count = 0
        for step in range(start_step, end_step + 1):
            total += self.get_current_intensity(region, step)
            count += 1
        
        return total / count if count > 0 else 0.0
