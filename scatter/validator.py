"""Validation of the input settings."""
from typing import Any, Dict


class ValidateLoad:
    """Validation of the loading settings."""

    @staticmethod
    def __validate_pulse_load(pulse_load_dict: Dict[str, Any]) -> None:
        """Validate the settings of a pulse load (not implemented yet).

        Args:
            pulse_load_dict: loading settings.
        """

    @staticmethod
    def __validate_heaviside_load(heaviside_load_dict: Dict[str, Any]) -> None:
        """Validate the settings of a heaviside load (not implemented yet).

        Args:
            heaviside_load_dict: loading settings.
        """

    @staticmethod
    def __validate_moving_load(moving_load_dict: Dict[str, Any]) -> None:
        """Validate the settings of a moving load (not implemented yet).

        Args:
            moving_load_dict: loading settings.
        """

    @staticmethod
    def __validate_moving_at_plane_load(moving_at_plane_load_dict: Dict[str, Any]) -> None:
        """Validate the settings of a moving load on a plane (not implemented yet).

        Args:
            moving_at_plane_load_dict: loading settings.
        """

    @staticmethod
    def __validate_rose_load(rose_load_dict: Dict[str, Any]) -> None:
        """Validate the settings of a rose load (not implemented yet).

        Args:
            rose_load_dict: loading settings.
        """

    @staticmethod
    def validate(loading: Dict[str, Any]) -> None:
        """Validate the loading settings, and set the default values of the optional settings.

        Args:
            loading: loading settings; "ini_steps" defaults to 5.

        Raises:
            AssertionError: if the loading settings have no "type".
            Exception: if the load type is not supported.
        """
        assert "type" in loading

        if loading["type"] == "pulse":
            ValidateLoad.__validate_pulse_load(loading)
        elif loading["type"] == "heaviside":
            ValidateLoad.__validate_heaviside_load(loading)
        elif loading["type"] == "moving":
            ValidateLoad.__validate_moving_load(loading)
        elif loading["type"] == "moving_at_plane":
            ValidateLoad.__validate_moving_at_plane_load(loading)
        elif loading["type"] == "rose":
            ValidateLoad.__validate_rose_load(loading)
        else:
            raise Exception(f'Error: Load type {loading["type"]} not supported')

        loading.setdefault("ini_steps", 5)
