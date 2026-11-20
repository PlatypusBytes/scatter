"""Random fields of material properties."""
import os
from typing import Any, Dict, List, Union

import meshio
import numpy as np
import numpy.typing as npt
from gstools import SRF, Exponential, Gaussian, Linear, Matern

from scatter.mesher import ReadMesh

# gstools covariance model per scatter model name
COVARIANCE_MODELS = {"Gaussian": Gaussian, "Exponential": Exponential, "Matern": Matern, "Linear": Linear}

# meshio cell type per scatter element type
MESHIO_ELEMENT_TYPES = {
    "hexa8": "hexahedron",
    "hexa20": "hexahedron20",
    "tetra4": "tetra",
    "tetra10": "tetra10",
    "tri3": "triangle",
    "tri6": "triangle6",
    "quad4": "quad",
    "quad8": "quad8",
}


class RF:
    """Random field generator for material properties."""

    def __init__(self, random_properties: Dict[str, Dict[str, Any]], materials: Dict[str, Dict[str, float]],
                 output_folder: str, element_type: str) -> None:
        """Initialise the random field generator, and create the output folder.

        Args:
            random_properties: random field properties per material name: "key_material" (the material property),
                "model_name", "theta" [m], "aniso_x" [-], "aniso_z" [-], "seed_number" and "std_value".
            materials: material properties per material name.
            output_folder: folder where the random field properties are written.
            element_type: element type of the mesh, e.g. hexa8 or tetra4.
        """
        self.random_properties = random_properties
        self.materials = materials
        self.output_folder = output_folder
        self.element_type = element_type
        self.lognormal = True
        self.new_material: Dict[str, Dict[str, float]] = {}

        os.makedirs(output_folder, exist_ok=True)

    def element_type_to_meshio_element_type(self) -> str:
        """Translate the scatter element type to the meshio cell type.

        Returns:
            meshio cell type.
        """
        return MESHIO_ELEMENT_TYPES[self.element_type]

    def generate_random_fields(self, model: ReadMesh, ndim: int, angles: Union[float, List[float]]) -> None:
        """Generate a random field for every random field material, with one material per element of the field.

        The material list and material index of the model, and the materials dictionary, are updated in place.

        Args:
            model: scatter model.
            ndim: dimension of the model.
            angles: rotation angles of the random field [rad].
        """
        rf_names = set(self.random_properties.keys())

        # start from a clean, globally unique material indexing
        new_model_materials = []
        new_materials_index = np.array(model.materials_index, dtype=int).copy()
        counter = 0

        # keep the materials that are not part of a random field and re-index them
        for material in model.materials:
            name = material[2]
            if name in rf_names:
                continue
            new_model_materials.append([material[0], counter, name])
            new_materials_index[model.materials_index == material[1]] = counter
            counter += 1

        for material in model.materials:
            name = material[2]
            if name not in rf_names:
                continue
            props = self.random_properties[name]

            # elements belonging to this material
            mask = model.materials_index == material[1]
            element_positions = np.where(mask)[0]

            # mean value of the property from the base material
            mean = self.materials[name][props["key_material"]]
            fields = self.generate_gstools_rf(model.nodes, model.elem[mask], ndim, props, mean, angles=angles)

            # create a unique material for every element of the random field
            for i, pos in enumerate(element_positions):
                vals = dict(self.materials[name])
                vals[props["key_material"]] = fields[i]

                new_name = f"{name}_rf_{i + 1}"
                self.new_material[new_name] = vals
                new_model_materials.append([material[0], counter, new_name])
                new_materials_index[pos] = counter
                counter += 1

        model.materials = new_model_materials
        model.materials_index = new_materials_index
        self.materials.update(self.new_material)

    def generate_gstools_rf(self,
                            nodes: npt.NDArray[np.float64],
                            elements: npt.NDArray[np.int64],
                            ndim: int,
                            props: Dict[str, Any],
                            mean: float,
                            angles: Union[float, List[float]] = 0.0) -> npt.NDArray[np.float64]:
        """Generate a random field of one material with gstools, at the element centroids.

        Args:
            nodes: node number and coordinates of each node [m].
            elements: node numbers of each element of the material.
            ndim: dimension of the model.
            props: random field properties of the material.
            mean: mean value of the property.
            angles: rotation angles of the random field [rad].

        Returns:
            Value of the property in each element.

        Raises:
            ValueError: if the covariance model is not supported.
        """
        seed = abs(props["seed_number"])
        len_scale = np.array([props["aniso_x"], 1, props["aniso_z"]]) * props["theta"]

        if self.lognormal:
            var = np.log((props["std_value"] / mean)**2 + 1)
            mean = np.log(mean**2 / (np.sqrt(mean**2 + props["std_value"]**2)))
        else:
            var = props["std_value"]**2

        model_name = props["model_name"]
        if model_name not in COVARIANCE_MODELS:
            raise ValueError(f'model name: "{model_name}" is not supported')
        model = COVARIANCE_MODELS[model_name](dim=ndim, var=var, len_scale=len_scale, angles=angles)
        srf = SRF(model, mean=mean, seed=seed)

        mesh = meshio.Mesh(nodes[:, 1:], {self.element_type_to_meshio_element_type(): elements - 1})
        srf.mesh(mesh, points="centroids", name="c-field-0", seed=seed)

        field: npt.NDArray[np.float64] = list(mesh.cell_data.values())[0][0]
        if self.lognormal:
            field = np.exp(field)
        return field

    def dump(self) -> None:
        """Write the random field properties to rf_props.txt in the output folder."""
        with open(os.path.join(self.output_folder, 'rf_props.txt'), 'w') as fo:
            fo.write('Random field properties\n')
            for name, props in self.random_properties.items():
                fo.write(f"\nMaterial: {name}\n")
                fo.write(f"Model: {props['model_name']}\n")
                fo.write('Theta: ' + str(props['theta']) + '\n')
                fo.write('Aniso_x: ' + str(props['aniso_x']) + '\n')
                fo.write('Aniso_z: ' + str(props['aniso_z']) + '\n')
                fo.write('Seed number: ' + str(props['seed_number']) + '\n')
                fo.write('Mean value: ' + str(self.materials[name][props['key_material']]) + '\n')
                fo.write('Std value: ' + str(props['std_value']) + '\n')
                fo.write('Log normal: ' + str(self.lognormal) + '\n')
