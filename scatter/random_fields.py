import os

import meshio
import numpy as np
from gstools import SRF, Exponential, Gaussian, Linear, Matern


class RF:
    def __init__(self, random_properties, materials, output_folder, element_type):
        """
        Initializes the random field generator with the given properties, materials, output folder, and element type.

        :param random_properties: dictionary with random field properties
        :param materials: dictionary with material properties
        :param output_folder: location of the output folder
        :param element_type: type of the finite element (e.g., "hexa8", "tetra4", etc.)
        """
        self.random_properties = random_properties
        self.materials = materials
        self.output_folder = output_folder
        self.element_type = element_type
        self.lognormal = True
        self.new_material = {}

        os.makedirs(output_folder, exist_ok=True)


    def element_type_to_meshio_element_type(self):
        """
        Translates scatter element types to meshio element types
        """
        translation_dict ={"hexa8": "hexahedron",
                           "hexa20": "hexahedron20",
                           "tetra4": "tetra",
                           "tetra10": "tetra10",
                           "tri3": "triangle",
                           "tri6":"triangle6",
                           "quad4": "quad",
                           "quad8": "quad8"}

        return translation_dict[self.element_type]

    def generate_random_fields(self, model, ndim, angles):
        """
        Generates a random field for every material defined in the random properties and
        rewrites the material list of the model so that every element belonging to a random
        field material gets its own material entry.

        :param model: the finite element model
        :param ndim: number of dimensions of the model
        :param angles: rotation angles for the random field
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

        # generate the random field for each random field material
        for material in model.materials:
            name = material[2]
            if name not in rf_names:
                continue

            props = self.random_properties[name]

            # elements belonging to this material
            mask = model.materials_index == material[1]
            elements = model.elem[mask]
            element_positions = np.where(mask)[0]

            # mean value of the property from the base material
            mean = self.materials[name][props["key_material"]]

            # generate the field for these elements
            fields = self.generate_gstools_rf(model.nodes, elements, ndim, props, mean, angles=angles)

            # create a unique material for every element of the random field
            for i, pos in enumerate(element_positions):
                vals = dict(self.materials[name])
                vals[props["key_material"]] = fields[i]

                new_name = f"{name}_rf_{i + 1}"
                self.new_material[new_name] = vals
                new_model_materials.append([material[0], counter, new_name])
                new_materials_index[pos] = counter
                counter += 1

        # update the model and the materials dictionary
        model.materials = new_model_materials
        model.materials_index = new_materials_index
        self.materials.update(self.new_material)


    def generate_gstools_rf(self, nodes, elements, ndim, props, mean, angles=0.0):
        """
        Generates a random field with the gstools random field generator for a single material
        and returns the field value for every element.

        :param nodes: array of node coordinates
        :param elements: array of element connectivities
        :param ndim: number of dimensions of the model
        :param props: dictionary with random field properties for the material
        :param mean: mean value of the property for the material
        :param angles: rotation angles for the random field (default: 0.0)
        :return: array of random field values for every element
        """

        # make sure seed is positive
        seed = abs(props["seed_number"])

        # set scale of fluctuation
        len_scale = np.array([props["aniso_x"], 1, props["aniso_z"]]) * props["theta"]

        # calculate variance and mean
        if self.lognormal:
            var = np.log((props["std_value"] / mean) ** 2 + 1)
            mean = np.log(mean ** 2 / (np.sqrt(mean ** 2 + props["std_value"] ** 2)))
        else:
            var = props["std_value"] ** 2

        model_name = props["model_name"]

        # initialise model
        if model_name == 'Gaussian':
            model = Gaussian(dim=ndim, var=var, len_scale=len_scale, angles=angles)
        elif model_name == 'Exponential':
            model = Exponential(dim=ndim, var=var, len_scale=len_scale, angles=angles)
        elif model_name == 'Matern':
            model = Matern(dim=ndim, var=var, len_scale=len_scale, angles=angles)
        elif model_name == 'Linear':
            model = Linear(dim=ndim, var=var, len_scale=len_scale, angles=angles)
        else:
            print('model name: "', model_name, '" is not supported')
            return

        # initialise random field
        srf = SRF(model, mean=mean, seed=seed)

        # create meshio mesh
        element_type = self.element_type_to_meshio_element_type()
        mesh = meshio.Mesh(nodes[:, 1:], {element_type: elements - 1})

        # create random field
        srf.mesh(mesh, points="centroids", name="c-field-0", seed=seed)

        # get random field values per element
        field = list(mesh.cell_data.values())[0][0]
        if self.lognormal:
            field = np.exp(field)

        return field

    def dump(self):
        """
        Dumps the random field properties to a text file in the output folder.
        """
        # dump information about the random fields
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
