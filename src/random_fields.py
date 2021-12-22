import os
import numpy as np
import ctypes as ct
from gstools import SRF, Exponential, Gaussian, Matern, Linear
import meshio


class RF:
    def __init__(self, random_properties, materials, output_folder, element_type):
        self.theta = random_properties["theta"]
        self.seed = random_properties["seed_number"]
        self.materials = materials
        self.material_name = random_properties["material"]
        self.key_material = random_properties["key_material"]
        self.sd = random_properties["std_value"]
        self.aniso_x = random_properties["aniso_x"]
        self.aniso_z = random_properties["aniso_z"]
        self.lognormal = True
        self.new_material = {}
        self.new_model_material = []
        self.new_material_index = []
        self.output_folder = output_folder
        self.element_type = element_type

        # check if output folder exists. if not creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return

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

    def generate_gstools_rf(self, nodes, elements, ndim, angles=0.0, model_name='Exponential'):
        """
        Generates a random field with the gstools random field generator
        """

        # make sure seed is positive
        seed = abs(self.seed)

        # set scale of fluctuation
        len_scale = np.array([self.aniso_x, 1, self.aniso_z])*self.theta

        # calculate variance and mean
        mean = self.materials[self.material_name][self.key_material]

        if self.lognormal:
            var = np.log((self.sd/mean)**2 + 1)
            mean = np.log(mean**2/(np.sqrt(mean**2 + self.sd**2)))
        else:
            var = self.sd**2

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
        srf.mesh(mesh, points="centroids", name="c-field-{}".format(0), seed=seed+0)

        # get random fields
        if self.lognormal:
            self.fields = [np.exp(field[0]) for field in mesh.cell_data.values()]
        else:
            self.fields = [field[0] for field in mesh.cell_data.values()]

        # rewrite material dictionary
        for idx in range(len(elements)):
            vals = dict(self.materials[self.material_name])
            vals[self.key_material] = self.fields[0][idx]

            # update new material dictionary
            self.new_material.update({f"material_{str(idx + 1)}": vals})
            # update model material
            self.new_model_material.append([3, idx, f"material_{str(idx + 1)}"])
            # update material index
            self.new_material_index.append(int(idx))

        return srf

    def dump(self):
        # dump information about the RF
        with open(os.path.join(self.output_folder, 'rf_props.txt'), 'w') as fo:
            fo.write('Random field properties\n')
            fo.write('Theta: ' + str(self.theta) + '\n')
            fo.write('Aniso_x: ' + str(self.aniso_x) + '\n')
            fo.write('Aniso_z: ' + str(self.aniso_z) + '\n')
            fo.write('Seed number: ' + str(self.seed) + '\n')
            fo.write('Mean value: ' + str(self.materials[self.material_name][self.key_material]) + '\n')
            fo.write('Std value: ' + str(self.sd) + '\n')
            fo.write('Log normal: ' + str(self.lognormal) + '\n')
        return

