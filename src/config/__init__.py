"""Config package – re-exports the settings singleton."""

from .settings import Settings, load_settings, settings

__all__ = ["Settings", "load_settings", "settings"]
