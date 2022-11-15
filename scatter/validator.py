from typing import Dict


class ValidateLoad:
    def __init__(self):
        pass

    @staticmethod
    def __validate_pulse_load(pulse_load_dict: Dict):
        # todo validate load
        pass

    @staticmethod
    def __validate_heaviside_load(heaviside_load_dict: Dict):
        # todo validate load
        pass

    @staticmethod
    def __validate_moving_load(moving_load_dict: Dict):
        # todo validate load
        pass

    @staticmethod
    def __validate_moving_at_plane_load(moving_at_plane_load_dict: Dict):
        # todo validate load
        pass

    @staticmethod
    def __validate_rose_load(rose_load_dict: Dict):
        # todo validate load
        pass

    @staticmethod
    def validate(loading: Dict):
        """
        Validates dictionary with loading settings, and sets default values if settings is optional

        :param loading: dictionary with loading settings
        """

        assert "type" in loading

        if "type" == "pulse":
            ValidateLoad.__validate_pulse_load(loading)
        elif "type" == "heaviside":
            ValidateLoad.__validate_heaviside_load(loading)
        elif "type" == "moving":
            ValidateLoad.__validate_moving_load(loading)
        elif "type" == "moving_at_plane":
            ValidateLoad.__validate_moving_at_plane_load(loading)
        elif "type" == "rose":
            ValidateLoad.__validate_rose_load(loading)
        else:
            Exception(f'Error: Load type {loading["type"]} not supported')

        # fill in default values
        loading.setdefault("ini_steps", 5)

