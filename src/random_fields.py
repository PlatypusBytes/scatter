import os
import numpy as np
import ctypes as ct
from gstools import SRF, Exponential, Gaussian, Matern, Linear
import meshio


class RF:
    def __init__(self, random_properties, materials, output_folder, element_type):
        self.n = random_properties["number_realisations"]  # number of realisations in one set
        self.max_lvl = []  # number of levels of subdivision (2**max_lvl) is size.
        self.cellsize = random_properties["element_size"]  # Cell size
        self.theta = random_properties["theta"]
        self.xcells = []  # Number of cells x dir
        self.ycells = []  # Number of cells x dir
        self.zcells = []  # Number of cells x dir
        self.seed = random_properties["seed_number"]
        self.materials = materials
        self.material_name = random_properties["material"]
        self.key_material = random_properties["key_material"]
        # self.mean = random_properties["material"][random_properties["index_material"]]
        self.sd = random_properties["std_value"]
        self.aniso_x = random_properties["aniso_x"]
        self.aniso_z = random_properties["aniso_z"]
        self.lognormal = True
        self.fieldfromcentre = False
        self.fields = []
        self.element_index = []
        self.new_material = {}
        self.new_model_material = []
        self.new_material_index = []
        self.new_elements = []
        self.output_folder = output_folder
        self.tol = 1e-6  # tolerance for checking equality
        self.element_type = element_type

        # check if output folder exists. if not creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return

    def element_type_to_meshio_element_type(self):
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
        len_scale = np.array([self.aniso_x, 1, self.aniso_y])*self.theta

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
        # todo, make dependent on type of element, currently only hexahedron elements are generated

        element_type = self.element_type_to_meshio_element_type()
        mesh = meshio.Mesh(nodes[:, 1:], {element_type: elements - 1})

        # create random fields
        for i in range(self.n):
            srf.mesh(mesh, points="centroids", name="c-field-{}".format(i), seed=seed+i)

        # get random fields
        if self.lognormal:
            self.fields = [np.exp(field[0]) for field in mesh.cell_data.values()]
        else:
            self.fields = [field[0] for field in mesh.cell_data.values()]

        # rewrite material dictionary
        for idx in range(len(elements)):
            vals = dict(self.materials[self.material_name])
            # ToDo: work out the index 0 for Monte Carlo analysis
            vals[self.key_material] = self.fields[0][idx]

            # update new material dictionary
            self.new_material.update({f"material_{str(idx + 1)}": vals})
            # update model material
            self.new_model_material.append([3, idx, f"material_{str(idx + 1)}"])
            # update material index
            self.new_material_index.append(int(idx))

        return srf

    def generate(self, nodes, elements):

        # determine number of cells on each direction
        self.xcells = int((np.max(nodes[:, 1]) - np.min(nodes[:, 1])) / self.cellsize)
        self.ycells = int((np.max(nodes[:, 2]) - np.min(nodes[:, 2])) / self.cellsize)
        self.zcells = int((np.max(nodes[:, 3]) - np.min(nodes[:, 3])) / self.cellsize)

        # compute middle point of RF mesh
        mean_coord_RF = []
        idx_coords = []
        for y in range(self.ycells):
            for z in range(self.zcells):
                for x in range(self.xcells):
                    x_mean = (range(self.xcells + 1)[x] + range(self.xcells + 1)[x + 1]) / 2 * self.cellsize
                    y_mean = (range(self.ycells + 1)[y] + range(self.ycells + 1)[y + 1]) / 2 * self.cellsize
                    z_mean = (range(self.zcells + 1)[z] + range(self.zcells + 1)[z + 1]) / 2 * self.cellsize
                    mean_coord_RF.append([x_mean, y_mean, z_mean])
                    idx_coords.append([x, z, y])

        # maximum number of cells
        max_nb_cells = np.max([self.xcells, self.ycells, self.zcells]) + 16

        self.max_lvl = int(np.ceil(np.log(max_nb_cells) / np.log(2)))

        # generate random field
        self.fields = rand3d(self.n, self.max_lvl, self.cellsize, self.theta, self.xcells, self.zcells, self.ycells,
                             self.seed, self.materials[self.material_name][self.key_material], self.sd,
                             self.lognormal, self.fieldfromcentre, anisox=self.aniso_x, anisoy=self.aniso_y)

        # remap fields into a list with materials according to the elements
        for j, el in enumerate(elements):
            # nodes in element
            nod = el
            # coordinates nodes in element
            # coord_nod = nodes[nod - 1]  # [nodes[n - 1] for n in nod]
            idx_nodes = [np.where(n == nodes[:, 0].astype(int))[0][0] for n in nod]
            coord_nod = nodes[idx_nodes]

            # compute middle point
            middle_point = np.array([np.mean(coord_nod[:, 1]),
                                     np.mean(coord_nod[:, 2]),
                                     np.mean(coord_nod[:, 3])])

            # find index of middle point element in RF mesh
            idx = [i for i, val in enumerate(np.abs(middle_point - np.array(mean_coord_RF)) <= self.tol) if all(val)][0]

            # rewrite material dictionary
            vals = dict(self.materials[self.material_name])
            # ToDo: work out the index 0 for Monte Carlo analysis
            vals[self.key_material] = self.fields[0][idx_coords[idx][0],
                                                     idx_coords[idx][1],
                                                     idx_coords[idx][2]]

            # update new material dictionary
            self.new_material.update({f"material_{str(j + 1)}": vals})
            # update model material
            self.new_model_material.append([3, j, f"material_{str(j + 1)}"])
            # update material index
            self.new_material_index.append(int(j))

        return

    def dump(self):
        # dump information about the RF
        with open(os.path.join(self.output_folder, 'rf_props.txt'), 'w') as fo:
            fo.write('Random field properties\n')
            fo.write('Number of subdivisions: ' + str(self.max_lvl) + '\n')
            fo.write('Cell size: ' + str(self.cellsize) + '\n')
            fo.write('Theta: ' + str(self.theta) + '\n')
            fo.write('Seed number: ' + str(self.seed) + '\n')
            fo.write('Mean value: ' + str(self.materials[self.material_name][self.key_material]) + '\n')
            fo.write('Std value: ' + str(self.sd) + '\n')
            fo.write('Log normal: ' + str(self.lognormal) + '\n')
            fo.write('Field from centre: ' + str(self.fieldfromcentre) + '\n')
        return


def rand3d(n, max_lvl, cellsize, theta, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre, anisox=1, anisoy=1):

    DLL = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs', 'RF3D.dll'))

    initialiseblackbox = DLL.RAND3DMOD_mp_INITIALISEBLACKBOX
    blackbox3d = DLL.RAND3DMOD_mp_BLACKBOX3D

    max_lvl_ptr = ct.pointer(ct.c_int(max_lvl))
    level = max_lvl
    level_ptr = ct.pointer(ct.c_int(level))

    squash = 1
    squash_ptr = ct.pointer(ct.c_int(squash))
    stretchx = 1
    stretchx_ptr = ct.pointer(ct.c_int(stretchx))
    stretchy = 1
    stretchy_ptr = ct.pointer(ct.c_int(stretchy))

    cellsize_ptr = ct.pointer(ct.c_double(cellsize))

    theta_ptr = ct.pointer(ct.c_double(theta))

    xcells_ptr = ct.pointer(ct.c_int(xcells))

    ycells_ptr = ct.pointer(ct.c_int(ycells))

    zcells_ptr = ct.pointer(ct.c_int(zcells))

    seed_ptr = ct.pointer(ct.c_int(seed))

    meantop = mean
    meantop_ptr = ct.pointer(ct.c_double(meantop))
    sdtop = sd
    sdtop_ptr = ct.pointer(ct.c_double(sdtop))
    meanbot = mean
    meanbot_ptr = ct.pointer(ct.c_double(meanbot))
    sdbot = sd
    sdbot_ptr = ct.pointer(ct.c_double(sdbot))

    anisox_ptr = ct.pointer(ct.c_int(anisox))
    anisoy_ptr = ct.pointer(ct.c_int(anisoy))

    lognormal_ptr = ct.pointer(ct.c_bool(lognormal))

    fieldfromcentre_ptr = ct.pointer(ct.c_bool(fieldfromcentre))

    field = np.zeros([zcells,ycells,xcells])
    field_ptr = np.ctypeslib.as_ctypes(field)

    a_27c = np.zeros([max_lvl, 27, 7])
    a_27c_ptr = np.ctypeslib.as_ctypes(a_27c)

    c_27c = np.zeros([max_lvl, 28])
    c_27c_ptr = np.ctypeslib.as_ctypes(c_27c)

    initialiseblackbox(max_lvl_ptr, cellsize_ptr, theta_ptr, a_27c_ptr, c_27c_ptr,
                       squash_ptr, stretchx_ptr, stretchy_ptr, anisox_ptr, anisoy_ptr,
                       xcells_ptr, ycells_ptr, zcells_ptr, level_ptr, fieldfromcentre_ptr)

    fields = []

    for realisation in range(n):
        blackbox3d(a_27c_ptr, c_27c_ptr, xcells_ptr, ycells_ptr, zcells_ptr, meantop_ptr, sdtop_ptr, meanbot_ptr,
                   sdbot_ptr, theta_ptr, cellsize_ptr, level_ptr, seed_ptr, field_ptr, squash_ptr, stretchx_ptr,
                   stretchy_ptr, lognormal_ptr, fieldfromcentre_ptr)
        fields.append(np.transpose(np.array(field)))
        seed = seed - 1
    return fields
