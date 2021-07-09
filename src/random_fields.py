import os
import numpy as np
import ctypes as ct
from gstools import SRF, Exponential, Gaussian, Matern, Linear
import meshio


class RF:
    def __init__(self, random_properties, materials, output_folder):
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
        self.aniso_y = random_properties["aniso_y"]
        self.lognormal = True
        self.fieldfromcentre = False
        self.fields = []
        self.middle_point = []
        self.element_index = []
        self.new_material = {}
        self.new_model_material = []
        self.new_material_index = []
        self.new_elements = []
        self.output_folder = output_folder
        # check if output folder exists. if not creates
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        return

    def generate_gstools_rf(self, nodes, elements, ndim, angles=0.0, model_name='Gaussian'):
        """
        Generates a random field with the gstools random field generator
        """

        # make sure seed is positive
        seed = abs(self.seed)

        # set scale of fluctuation
        len_scale = np.array([self.aniso_x, self.aniso_y, 1])*self.theta

        # calculate variance
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
        srf = SRF(model, mean=self.materials[self.material_name][self.key_material], seed=seed)

        # create meshio mesh
        # todo, make dependent on type of element, currently only hexahedron elements are generated
        mesh = meshio.Mesh(nodes[:, 1:], {"hexahedron": elements - 1})

        # create random fields
        for i in range(self.n):
            srf.mesh(mesh, points="centroids", name="c-field-{}".format(i), seed=seed+i)

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

        # maximum number of cells
        max_nb_cells = np.max([self.xcells, self.ycells, self.zcells]) + 16

        self.max_lvl = int(np.ceil(np.log(max_nb_cells) / np.log(2)))

        # generate random field
        self.fields = rand3d(self.n, self.max_lvl, self.cellsize, self.theta, self.xcells, self.ycells, self.zcells,
                             self.seed, self.materials[self.material_name][self.key_material], self.sd,
                             self.lognormal, self.fieldfromcentre, anisox=self.aniso_x, anisoy=self.aniso_y)

        # remap fields into a list with materials according to the elements
        for el in elements:
            # nodes in element
            nod = el
            # coordinates nodes in element
            coord_nod = nodes[nod - 1]  # [nodes[n - 1] for n in nod]
            # compute middle point
            self.middle_point.append([np.mean(coord_nod[:, 1]).round(2),
                                      np.mean(coord_nod[:, 2]).round(2),
                                      np.mean(coord_nod[:, 3]).round(2)])
            self.element_index.append([int(np.mean(coord_nod[:, 1]).round(2) / self.cellsize),
                                       int(np.mean(coord_nod[:, 2]).round(2) / self.cellsize),
                                       int(np.mean(coord_nod[:, 3]).round(2) / self.cellsize)])

        # rewrite material dictionary
        for idx in range(len(elements)):
            vals = dict(self.materials[self.material_name])
            # ToDo: work out the index 0 for Monte Carlo analysis
            vals[self.key_material] = self.fields[0][self.element_index[idx][0],
                                                     self.element_index[idx][1],
                                                     self.element_index[idx][2]]

            # update new material dictionary
            self.new_material.update({f"material_{str(idx + 1)}": vals})
            # update model material
            self.new_model_material.append([3, idx, f"material_{str(idx + 1)}"])
            # update material index
            self.new_material_index.append(int(idx))

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
