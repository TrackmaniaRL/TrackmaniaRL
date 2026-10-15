"""Offline, fail-closed preparation and analysis of Trackmania experiments."""

from trackmaniarl.research.manifest import load_manifest, preflight
from trackmaniarl.research.results import aggregate_registry

__all__ = ["aggregate_registry", "load_manifest", "preflight"]
