"""
WattTime API carbon data provider.

Provides real-time carbon intensity data from the WattTime API,
which covers grid operators primarily in the United States.

Requires WattTime API credentials. Set environment variables:
- WATTTIME_USERNAME
- WATTTIME_PASSWORD

For API documentation, see: https://www.watttime.org/api-documentation/
"""

import os
import time
from typing import Dict, List, Optional
from datetime import datetime, timezone

from server.providers.base import CarbonProvider
from server.providers.config import get_provider_region, DEFAULT_CARBON_INTENSITY

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


class WattTimeProvider(CarbonProvider):
    """
    Carbon provider using the WattTime API.
    
    WattTime provides real-time and historical carbon intensity data
    for grid operators, primarily in North America.
    
    The API requires authentication. Set credentials via environment
    variables or pass them to the constructor.
    
    Note: This provider makes HTTP requests and should be used with
    appropriate rate limiting and caching for production workloads.
    """

    BASE_URL = "https://api.watttime.org"
    
    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        cache_ttl: int = 300,
    ):
        """
        Initialize WattTime provider.
        
        Args:
            username: WattTime API username. Defaults to WATTTIME_USERNAME env var
            password: WattTime API password. Defaults to WATTTIME_PASSWORD env var
            cache_ttl: Cache time-to-live in seconds (default 5 minutes)
        """
        self._username = username or os.environ.get("WATTTIME_USERNAME", "")
        self._password = password or os.environ.get("WATTTIME_PASSWORD", "")
        self._token: Optional[str] = None
        self._token_expires: float = 0
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple] = {}
        
        self._supported_regions = [
            "us-east-1", "us-east-2", "us-west-1", "us-west-2",
            "us-central1",
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

    async def _get_token(self) -> str:
        """Get or refresh API token."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp required for WattTime API. Install with: pip install aiohttp")
        
        if self._token and time.time() < self._token_expires:
            return self._token
        
        async with aiohttp.ClientSession() as session:
            auth = aiohttp.BasicAuth(self._username, self._password)
            async with session.get(f"{self.BASE_URL}/login", auth=auth) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    self._token = data.get("token", "")
                    self._token_expires = time.time() + 1800
                    return self._token
                else:
                    raise RuntimeError(f"WattTime auth failed: {resp.status}")
        
        return ""

    async def _fetch_index(self, region: str) -> Optional[Dict]:
        """Fetch current carbon index for a region."""
        watttime_region = get_provider_region(region, "watttime")
        if not watttime_region:
            return None
        
        if not AIOHTTP_AVAILABLE:
            return None
        
        try:
            token = await self._get_token()
            headers = {"Authorization": f"Bearer {token}"}
            
            async with aiohttp.ClientSession() as session:
                params = {"ba": watttime_region}
                async with session.get(
                    f"{self.BASE_URL}/v3/signal-index",
                    headers=headers,
                    params=params
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception:
            pass
        
        return None

    def get_current_intensity(self, region: str, step: int) -> float:
        """
        Get current carbon intensity for a region.
        
        Note: In synchronous context, returns cached or default value.
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
        
        WattTime provides forecasts via their API. In sync context,
        returns current intensity repeated.
        """
        current = self.get_current_intensity(region, step)
        return [current] * lookahead

    def get_spot_multiplier(self, region: str, step: int) -> float:
        """
        Get spot price multiplier.
        
        WattTime doesn't provide pricing data. Returns default multiplier.
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

    async def fetch_current_intensity_async(self, region: str) -> float:
        """
        Asynchronously fetch current carbon intensity.
        
        This makes an actual API call to WattTime.
        """
        data = await self._fetch_index(region)
        if data and "moer" in data:
            intensity = float(data["moer"]) * 10
            self.update_intensity(region, intensity)
            return intensity
        
        return DEFAULT_CARBON_INTENSITY

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
