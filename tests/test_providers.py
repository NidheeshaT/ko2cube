"""
Tests for carbon data providers.
"""
import pytest
from typing import List

from server.providers.base import CarbonProvider, CarbonReading
from server.providers.config import (
    REGION_MAPPINGS, get_provider_region, get_all_mapped_regions,
    is_region_supported, get_region_profile,
)
from server.providers.static import StaticCarbonProvider
from server.providers.watttime import WattTimeProvider
from server.providers.electricity_map import ElectricityMapProvider


class TestCarbonReading:
    """Tests for CarbonReading dataclass."""

    def test_carbon_reading_creation(self):
        """CarbonReading can be created with required fields."""
        reading = CarbonReading(
            region="us-east-1",
            intensity_gco2_kwh=300.0,
            timestamp="step_5",
        )
        assert reading.region == "us-east-1"
        assert reading.intensity_gco2_kwh == 300.0
        assert reading.spot_multiplier == 1.0

    def test_carbon_reading_with_forecast(self):
        """CarbonReading can include forecast."""
        reading = CarbonReading(
            region="us-west-2",
            intensity_gco2_kwh=200.0,
            timestamp="step_0",
            forecast=[200.0, 180.0, 160.0],
            spot_multiplier=0.6,
        )
        assert len(reading.forecast) == 3
        assert reading.spot_multiplier == 0.6


class TestStaticCarbonProvider:
    """Tests for StaticCarbonProvider."""

    @pytest.fixture
    def provider(self) -> StaticCarbonProvider:
        """Create a StaticCarbonProvider using the default CSV."""
        return StaticCarbonProvider()

    def test_provider_loads_data(self, provider: StaticCarbonProvider):
        """Provider loads CSV data on initialization."""
        assert provider.num_steps > 0

    def test_provider_detects_regions(self, provider: StaticCarbonProvider):
        """Provider auto-detects regions from CSV."""
        regions = provider.get_supported_regions()
        assert len(regions) > 0
        assert "us-east-1" in regions

    def test_get_current_intensity(self, provider: StaticCarbonProvider):
        """Can get current carbon intensity for a region."""
        intensity = provider.get_current_intensity("us-east-1", step=0)
        assert intensity > 0

    def test_intensity_varies_by_step(self, provider: StaticCarbonProvider):
        """Carbon intensity changes across steps."""
        intensities = [
            provider.get_current_intensity("us-east-1", step=i)
            for i in range(10)
        ]
        assert len(set(intensities)) > 1

    def test_get_forecast(self, provider: StaticCarbonProvider):
        """Can get carbon intensity forecast."""
        forecast = provider.get_forecast("us-east-1", step=0, lookahead=6)
        assert len(forecast) == 6
        assert all(f > 0 for f in forecast)

    def test_forecast_first_equals_current(self, provider: StaticCarbonProvider):
        """First forecast value equals current intensity."""
        current = provider.get_current_intensity("us-east-1", step=5)
        forecast = provider.get_forecast("us-east-1", step=5, lookahead=6)
        assert forecast[0] == current

    def test_get_spot_multiplier(self, provider: StaticCarbonProvider):
        """Can get spot price multiplier."""
        multiplier = provider.get_spot_multiplier("us-east-1", step=0)
        assert 0 < multiplier < 2.0

    def test_validate_region(self, provider: StaticCarbonProvider):
        """Can validate if region is supported."""
        assert provider.validate_region("us-east-1") is True
        assert provider.validate_region("invalid-region") is False

    def test_get_reading(self, provider: StaticCarbonProvider):
        """Can get complete carbon reading."""
        reading = provider.get_reading("us-east-1", step=0, lookahead=6)
        assert isinstance(reading, CarbonReading)
        assert reading.region == "us-east-1"
        assert reading.intensity_gco2_kwh > 0
        assert len(reading.forecast) == 6

    def test_get_all_readings(self, provider: StaticCarbonProvider):
        """Can get readings for multiple regions."""
        regions = ["us-east-1", "us-west-2"]
        readings = provider.get_all_readings(regions, step=0)
        assert len(readings) == 2
        assert "us-east-1" in readings
        assert "us-west-2" in readings

    def test_get_min_intensity_region(self, provider: StaticCarbonProvider):
        """Can find region with minimum carbon intensity."""
        min_region = provider.get_min_intensity_region(step=0)
        assert min_region in provider.get_supported_regions()

    def test_get_average_intensity(self, provider: StaticCarbonProvider):
        """Can calculate average intensity over range."""
        avg = provider.get_average_intensity("us-east-1", start_step=0, end_step=5)
        assert avg > 0

    def test_step_wrapping(self, provider: StaticCarbonProvider):
        """Steps wrap around when exceeding data length."""
        num_steps = provider.num_steps
        i1 = provider.get_current_intensity("us-east-1", step=0)
        i2 = provider.get_current_intensity("us-east-1", step=num_steps)
        assert i1 == i2


class TestRegionConfig:
    """Tests for region configuration."""

    def test_region_mappings_not_empty(self):
        """Region mappings contain entries."""
        assert len(REGION_MAPPINGS) > 0

    def test_aws_regions_mapped(self):
        """AWS regions are mapped."""
        assert "us-east-1" in REGION_MAPPINGS
        assert "us-west-2" in REGION_MAPPINGS
        assert "eu-west-1" in REGION_MAPPINGS

    def test_mapping_has_providers(self):
        """Each mapping has provider-specific regions."""
        mapping = REGION_MAPPINGS["us-east-1"]
        assert "watttime" in mapping
        assert "electricity_map" in mapping

    def test_get_provider_region_valid(self):
        """Can get provider region for valid cloud region."""
        wt_region = get_provider_region("us-east-1", "watttime")
        assert wt_region == "PJM_WEST"

    def test_get_provider_region_invalid(self):
        """Returns None for unmapped regions."""
        result = get_provider_region("invalid-region", "watttime")
        assert result is None

    def test_get_all_mapped_regions(self):
        """Can get all regions for a provider."""
        wt_regions = get_all_mapped_regions("watttime")
        assert len(wt_regions) > 0
        assert "us-east-1" in wt_regions

    def test_is_region_supported(self):
        """Can check if region is supported."""
        assert is_region_supported("us-east-1") is True
        assert is_region_supported("fake-region") is False

    def test_get_region_profile(self):
        """Can get carbon intensity profile."""
        profile = get_region_profile("us-east-1")
        assert "min" in profile
        assert "max" in profile
        assert "avg" in profile
        assert profile["min"] < profile["max"]


class TestWattTimeProvider:
    """Tests for WattTimeProvider (without API calls)."""

    @pytest.fixture
    def provider(self) -> WattTimeProvider:
        """Create provider without credentials."""
        return WattTimeProvider()

    def test_supported_regions(self, provider: WattTimeProvider):
        """Provider lists supported regions."""
        regions = provider.get_supported_regions()
        assert len(regions) > 0
        assert "us-east-1" in regions

    def test_default_intensity(self, provider: WattTimeProvider):
        """Returns default intensity when no cache."""
        intensity = provider.get_current_intensity("us-east-1", step=0)
        assert intensity > 0

    def test_default_forecast(self, provider: WattTimeProvider):
        """Returns default forecast when no cache."""
        forecast = provider.get_forecast("us-east-1", step=0, lookahead=6)
        assert len(forecast) == 6

    def test_default_spot_multiplier(self, provider: WattTimeProvider):
        """Returns default spot multiplier."""
        multiplier = provider.get_spot_multiplier("us-east-1", step=0)
        assert multiplier == 0.5

    def test_update_intensity(self, provider: WattTimeProvider):
        """Can manually update cached intensity."""
        provider.update_intensity("us-east-1", 250.0)
        intensity = provider.get_current_intensity("us-east-1", step=0)
        assert intensity == 250.0


class TestElectricityMapProvider:
    """Tests for ElectricityMapProvider (without API calls)."""

    @pytest.fixture
    def provider(self) -> ElectricityMapProvider:
        """Create provider without API key."""
        return ElectricityMapProvider()

    def test_supported_regions(self, provider: ElectricityMapProvider):
        """Provider lists supported regions."""
        regions = provider.get_supported_regions()
        assert len(regions) > 0
        assert "eu-west-1" in regions

    def test_default_intensity(self, provider: ElectricityMapProvider):
        """Returns default intensity when no cache."""
        intensity = provider.get_current_intensity("eu-west-1", step=0)
        assert intensity > 0

    def test_default_forecast(self, provider: ElectricityMapProvider):
        """Returns default forecast when no cache."""
        forecast = provider.get_forecast("eu-west-1", step=0, lookahead=6)
        assert len(forecast) == 6

    def test_default_spot_multiplier(self, provider: ElectricityMapProvider):
        """Returns default spot multiplier."""
        multiplier = provider.get_spot_multiplier("eu-west-1", step=0)
        assert multiplier == 0.5

    def test_update_intensity(self, provider: ElectricityMapProvider):
        """Can manually update cached intensity."""
        provider.update_intensity("eu-west-1", 150.0)
        intensity = provider.get_current_intensity("eu-west-1", step=0)
        assert intensity == 150.0

    def test_update_forecast(self, provider: ElectricityMapProvider):
        """Can manually update cached forecast."""
        forecast_data = [100.0, 95.0, 90.0, 85.0, 80.0, 75.0]
        provider.update_forecast("eu-west-1", forecast_data)
        forecast = provider.get_forecast("eu-west-1", step=0, lookahead=6)
        assert forecast == forecast_data


class TestProviderInterface:
    """Tests to verify all providers implement the interface correctly."""

    @pytest.fixture(params=[
        StaticCarbonProvider,
        WattTimeProvider,
        ElectricityMapProvider,
    ])
    def provider(self, request) -> CarbonProvider:
        """Create each provider type."""
        return request.param()

    def test_is_carbon_provider(self, provider):
        """Provider is a CarbonProvider subclass."""
        assert isinstance(provider, CarbonProvider)

    def test_has_get_current_intensity(self, provider):
        """Provider has get_current_intensity method."""
        assert hasattr(provider, "get_current_intensity")
        result = provider.get_current_intensity("us-east-1", step=0)
        assert isinstance(result, (int, float))

    def test_has_get_forecast(self, provider):
        """Provider has get_forecast method."""
        assert hasattr(provider, "get_forecast")
        result = provider.get_forecast("us-east-1", step=0, lookahead=6)
        assert isinstance(result, list)
        assert len(result) == 6

    def test_has_get_spot_multiplier(self, provider):
        """Provider has get_spot_multiplier method."""
        assert hasattr(provider, "get_spot_multiplier")
        result = provider.get_spot_multiplier("us-east-1", step=0)
        assert isinstance(result, (int, float))

    def test_has_get_supported_regions(self, provider):
        """Provider has get_supported_regions method."""
        assert hasattr(provider, "get_supported_regions")
        result = provider.get_supported_regions()
        assert isinstance(result, list)
