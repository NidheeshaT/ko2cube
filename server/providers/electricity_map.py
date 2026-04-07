"""
Electricity Map API carbon data provider.

Provides real-time carbon intensity data from the Electricity Map API,
which covers regions globally.

Requires Electricity Map API key. Set environment variable:
- ELECTRICITY_MAP_API_KEY

For API documentation, see: https://static.electricitymaps.com/api/docs/index.html
"""

import os
import time
from typing import Dict, List, Optional

from server.providers.base import CarbonProvider
from server.providers.config import get_provider_region, DEFAULT_CARBON_INTENSITY

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class ElectricityMapProvider(CarbonProvider):
    """
    Carbon provider using the Electricity Map API.
    
    Electricity Map provides real-time carbon intensity data globally,
    with particularly good coverage in Europe.
    
    The API requires authentication via API key. Set via environment
    variable ELECTRICITY_MAP_API_KEY or pass to constructor.
    """

    BASE_URL = "https://api.electricitymap.org/v3"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize Electricity Map provider.
        
        Args:
            api_key: Electricity Map API key. Defaults to ELECTRICITY_MAP_API_KEY env var
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self._api_key = api_key or os.environ.get("ELECTRICITY_MAP_API_KEY", "")
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}
        
        self._supported_regions = [
            "us-east-1", "us-west-2", "eu-west-1", "eu-west-2",
            "eu-central-1", "eu-north-1", "ap-northeast-1",
        ]

    def _get_cached(self, key: str) -> Optional[float]:
        """Get value from cache if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return value
        return None

    def _set_cached(self, key: str, value: float) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, time.time())

    async def _fetch_carbon_intensity(self, region: str) -> Optional[Dict]:
        """Fetch current carbon intensity for a region from the API."""
        em_zone = get_provider_region(region, "electricity_map")
        if not em_zone:
            return None
        
        if not AIOHTTP_AVAILABLE:
            return None
        
        if not self._api_key:
            return None
        
        try:
            headers = {"auth-token": self._api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/carbon-intensity/latest",
                    headers=headers,
                    params={"zone": em_zone}
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        
        return None

    async def _fetch_forecast(self, region: str) -> Optional[List[Dict]]:
        """Fetch carbon intensity forecast for a region."""
        em_zone = get_provider_region(region, "electricity_map")
        if not em_zone:
            return None
        
        if not AIOHTTP_AVAILABLE:
            return None
        
        if not self._api_key:
            return None
        
        try:
            headers = {"auth-token": self._api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.BASE_URL}/carbon-intensity/forecast",
                    headers=headers,
                    params={"zone": em_zone}
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("forecast", [])
        except Exception:
            pass
        
        return None

    def get_current_intensity(self, region: str, step: int) -> float:
        """
        Get current carbon intensity for a region.
        
        In synchronous context, returns cached or default value.
        For real-time data, use async methods.
        """
        cache_key = f"intensity_{region}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        return DEFAULT_CARBON_INTENSITY

    def get_forecast(self, region: str, step: int, lookahead: int) -> List[float]:
        """
        Get carbon intensity forecast.
        
        Electricity Map provides 24-hour forecasts. In sync context,
        returns cached forecast or current intensity repeated.
        """
        cache_key = f"forecast_{region}"
        cached = self._get_cached(cache_key)
        if cached is not None and isinstance(cached, list):
            return cached[:lookahead] if len(cached) >= lookahead else cached + [cached[-1]] * (lookahead - len(cached))
        
        current = self.get_current_intensity(region, step)
        return [current] * lookahead

    def get_spot_multiplier(self, region: str, step: int) -> float:
        """
        Get spot price multiplier.
        
        Electricity Map doesn't provide pricing data. Returns default multiplier.
        """
        return 0.5

    def get_supported_regions(self) -> List[str]:
        """Return list of supported region identifiers."""
        return list(self._supported_regions)

    def update_intensity(self, region: str, intensity: float) -> None:
        """
        Manually update cached intensity value.
        
        Useful when fetching data asynchronously and updating the
        provider for synchronous access.
        """
        cache_key = f"intensity_{region}"
        self._set_cached(cache_key, intensity)

    def update_forecast(self, region: str, forecast: List[float]) -> None:
        """
        Manually update cached forecast value.
        """
        cache_key = f"forecast_{region}"
        self._cache[cache_key] = (forecast, time.time())

    async def fetch_current_intensity_async(self, region: str) -> float:
        """
        Asynchronously fetch current carbon intensity.
        
        This makes an actual API call to Electricity Map.
        """
        data = await self._fetch_carbon_intensity(region)
        if data and "carbonIntensity" in data:
            intensity = float(data["carbonIntensity"])
            self.update_intensity(region, intensity)
            return intensity
        
        return DEFAULT_CARBON_INTENSITY

    async def fetch_forecast_async(self, region: str, lookahead: int = 24) -> List[float]:
        """
        Asynchronously fetch carbon intensity forecast.
        """
        data = await self._fetch_forecast(region)
        if data:
            forecast = [
                float(entry.get("carbonIntensity", DEFAULT_CARBON_INTENSITY))
                for entry in data[:lookahead]
            ]
            if forecast:
                self.update_forecast(region, forecast)
                return forecast
        
        current = await self.fetch_current_intensity_async(region)
        return [current] * lookahead

    async def fetch_all_intensities_async(self, regions: List[str]) -> Dict[str, float]:
        """
        Asynchronously fetch carbon intensities for multiple regions.
        """
        results = {}
        for region in regions:
            try:
                results[region] = await self.fetch_current_intensity_async(region)
            except Exception:
                results[region] = DEFAULT_CARBON_INTENSITY
        return results
