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

        # fill in default values
        loading.setdefault("ini_steps", 5)


class ValidateMaterial:
    """
    Validates the material dictionary and normalises the ``formulation`` key.

    Each material is either ``dry`` (single phase, current behaviour) or ``biot``
    (saturated, two-phase, solved with the u-w formulation). The ``formulation``
    key is optional and defaults to ``dry`` for backward compatibility.
    """

    # required keys for a dry (single phase) material
    DRY_KEYS = ["density", "Young", "poisson"]

    # required keys for a Biot (u-w) material. Optional overrides that are not
    # validated here: biot_coefficient, biot_modulus, tortuosity, gravity, drag.
    BIOT_KEYS = ["porosity", "solid_density", "fluid_density", "Young", "poisson",
                 "solid_bulk_modulus", "fluid_bulk_modulus", "permeability"]

    @staticmethod
    def validate(materials: Dict):
        """
        Validates the material dictionary, sets the default formulation and checks
        that the required parameters are present.

        :param materials: dictionary with material properties
        """
        for name, props in materials.items():
            formulation = str(props.get("formulation", "dry")).lower()
            # normalise the formulation key in place
            props["formulation"] = formulation

            if formulation == "dry":
                required = ValidateMaterial.DRY_KEYS
            elif formulation == "biot":
                required = ValidateMaterial.BIOT_KEYS
            else:
                raise Exception(f"Error: material '{name}' has an unsupported formulation "
                                f"'{formulation}'. Use 'dry' or 'biot'.")

            missing = [key for key in required if key not in props]
            if missing:
                raise Exception(f"Error: material '{name}' ({formulation}) is missing the "
                                f"required parameter(s): {', '.join(missing)}")

