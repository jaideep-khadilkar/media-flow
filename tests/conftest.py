"""Pytest configuration and fixtures."""

import sys
from unittest.mock import MagicMock

# Mock ray module before any test imports to prevent decorator issues
mock_ray = MagicMock()
mock_ray.remote = lambda func: func  # Make decorator a no-op
sys.modules["ray"] = mock_ray
