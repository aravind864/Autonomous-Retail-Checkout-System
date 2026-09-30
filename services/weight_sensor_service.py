"""Weight sensor abstraction layer.

Provides a clean interface for weight sensor hardware.
Current implementation is mock/development only.
Future: implement HX711WeightSensorProvider.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class WeightSensorProvider(ABC):
    """Abstract base class for weight sensor hardware providers."""

    @abstractmethod
    def get_weight(self, sensor_id: str) -> Optional[float]:
        """Read current weight in grams from a sensor."""
        ...

    @abstractmethod
    def is_available(self, sensor_id: str) -> bool:
        """Check if a sensor is connected and operational."""
        ...

    @abstractmethod
    def calibrate(self, sensor_id: str, known_weight_g: float) -> bool:
        """Calibrate a sensor using a known reference weight."""
        ...


class MockWeightSensorProvider(WeightSensorProvider):
    """Development/test weight sensor that stores programmatic weight values.

    This provider does NOT generate automatic weight changes.
    It only responds to explicitly set values from the dev/test API.
    """

    def __init__(self):
        self._weights: Dict[str, float] = {}
        self._lock = threading.Lock()

    def get_weight(self, sensor_id: str) -> Optional[float]:
        """Return the last programmatically set weight for this sensor."""
        with self._lock:
            return self._weights.get(sensor_id)

    def set_weight(self, sensor_id: str, weight_g: float):
        """Programmatically set weight (for dev/test use only)."""
        with self._lock:
            self._weights[sensor_id] = weight_g
            logger.info(f"[MOCK SENSOR] Set {sensor_id} = {weight_g}g")

    def is_available(self, sensor_id: str) -> bool:
        """Mock sensors are always available."""
        return True

    def calibrate(self, sensor_id: str, known_weight_g: float) -> bool:
        """Mock calibration always succeeds."""
        logger.info(f"[MOCK SENSOR] Calibrated {sensor_id} at {known_weight_g}g")
        return True


# Placeholder for future hardware integration:
# class HX711WeightSensorProvider(WeightSensorProvider):
#     """Physical HX711 load cell sensor provider.
#
#     Requires GPIO access and physical hardware.
#     Not implemented — pending hardware integration.
#     """
#     def __init__(self, dout_pin: int, sck_pin: int):
#         ...
#     def get_weight(self, sensor_id: str) -> Optional[float]:
#         ...
#     def is_available(self, sensor_id: str) -> bool:
#         ...
#     def calibrate(self, sensor_id: str, known_weight_g: float) -> bool:
#         ...


# Module-level singleton
_provider: Optional[WeightSensorProvider] = None
_provider_lock = threading.Lock()


def get_sensor_provider() -> WeightSensorProvider:
    """Return the active sensor provider singleton (MockWeightSensorProvider in dev)."""
    global _provider
    with _provider_lock:
        if _provider is None:
            _provider = MockWeightSensorProvider()
            logger.info("[SENSOR] Using MockWeightSensorProvider (development mode)")
        return _provider


def set_sensor_provider(provider: WeightSensorProvider):
    """Replace the active sensor provider (for hardware integration)."""
    global _provider
    with _provider_lock:
        _provider = provider
        logger.info(f"[SENSOR] Sensor provider set to {type(provider).__name__}")
