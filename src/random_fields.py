class RF:

    def __init__(self, random_properties, output_folder):
        self.n = random_properties["number_realisations"]  # number of realisations in one set
        self.max_lvl = []  # number of levels of subdivision (2**max_lvl) is size.
        self.cellsize = random_properties["element_size"]  # Cell size
        self.theta = random_properties["theta"]
        self.xcells = []  # Number of cells x dir
        self.ycells = []  # Number of cells x dir
        self.zcells = []  # Number of cells x dir
        self.seed = random_properties["seed_number"]
        self.materials = random_properties["material"]
        self.index = random_properties["index_material"]
        # self.mean = random_properties["material"][random_properties["index_material"]]
        self.sd = random_properties["std_value"]
        self.lognormal = True
        self.fieldfromcentre = False
        self.fields = []
        self.middle_point = []
        self.element_index = []
        self.new_material = {}
        self.new_model_material = []
        self.new_elements = []
        self.output_folder = output_folder

        return

    def generate(self, nodes, elements):
        import numpy as np

        # determine number of cells on each direction
        self.xcells = int((np.max(nodes[:, 1]) - np.min(nodes[:, 1])) / self.cellsize)
        self.ycells = int((np.max(nodes[:, 2]) - np.min(nodes[:, 2])) / self.cellsize)
        self.zcells = int((np.max(nodes[:, 3]) - np.min(nodes[:, 3])) / self.cellsize)

        # maximum number of cells
        max_nb_cells = np.max([self.xcells, self.ycells, self.zcells]) + 16

        self.max_lvl = int(np.ceil(np.log(max_nb_cells) / np.log(2)))

        # generate random field
        self.fields = rand3d(self.n, self.max_lvl, self.cellsize, self.theta, self.xcells, self.ycells, self.zcells,
                             self.seed, self.materials[self.index], self.materials[self.index] * self.sd,
                             self.lognormal, self.fieldfromcentre)

        # remap fields into a list with materials according to the elements
        for el in elements:
            # nodes in element
            nod = el[5:]
            # coordinates nodes in element
            coord_nod = nodes[nod - 1]  # [nodes[n - 1] for n in nod]
            # compute middle point
            self.middle_point.append([np.mean(coord_nod[:, 1]).round(2),
                                      np.mean(coord_nod[:, 2]).round(2),
                                      np.mean(coord_nod[:, 3]).round(2)])
            self.element_index.append([int(np.mean(coord_nod[:, 1]).round(2) / self.cellsize),
                                       int(np.mean(coord_nod[:, 2]).round(2) / self.cellsize),
                                       int(np.mean(coord_nod[:, 3]).round(2) / self.cellsize)])

        self.new_elements = elements
        # rewrite material dictionary
        for idx in range(len(elements)):

            # property from the RF
            self.new_elements[idx][3] = idx

            vals = list(self.materials)
            # ToDo: work out the index 0 for Monte Carlo analysis
            vals[self.index] = self.fields[0][self.element_index[idx][0],
                                              self.element_index[idx][1],
                                              self.element_index[idx][2]]

            # update new material dictionary
            self.new_material.update({str(idx + 1): vals})
            # update model material
            self.new_model_material.append([3, idx, str(idx + 1)])
        return

    def dump(self):
        import os
        # dump information about the RF
        with open(os.path.join(self.output_folder, 'rf_props.txt'), 'w') as fo:
            fo.write('Random field properties\n')
            fo.write('Number of subdivisions: ' + str(self.max_lvl) + '\n')
            fo.write('Cell size: ' + str(self.cellsize) + '\n')
            fo.write('Theta: ' + str(self.theta) + '\n')
            fo.write('Seed number: ' + str(self.seed) + '\n')
            fo.write('Mean value: ' + str(self.materials[self.index]) + '\n')
            fo.write('Std value: ' + str(self.materials[self.index] * self.sd) + '\n')
            fo.write('Log normal: ' + str(self.lognormal) + '\n')
            fo.write('Field from centre: ' + str(self.fieldfromcentre) + '\n')
        return


def rand3d(n, max_lvl, cellsize, theta, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre, anisox=1, anisoy=1):
    import os
    import ctypes as ct
    import numpy as np

    DLL = ct.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'libs/RF3D.dll'))

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

    field = np.zeros([xcells,ycells,zcells])
    field_ptr = np.ctypeslib.as_ctypes(field)

    a_27c = np.zeros([7, 27, max_lvl])
    a_27c_ptr = np.ctypeslib.as_ctypes(a_27c)

    c_27c = np.zeros([28, max_lvl])
    c_27c_ptr = np.ctypeslib.as_ctypes(c_27c)

    initialiseblackbox(max_lvl_ptr, cellsize_ptr, theta_ptr, a_27c_ptr, c_27c_ptr,
                       squash_ptr, stretchx_ptr, stretchy_ptr, anisox_ptr, anisoy_ptr,
                       xcells_ptr, ycells_ptr, zcells_ptr, level_ptr, fieldfromcentre_ptr)

    fields = []

    for realisation in range(n):
        blackbox3d(a_27c_ptr, c_27c_ptr, xcells_ptr, ycells_ptr, zcells_ptr, meantop_ptr, sdtop_ptr, meanbot_ptr,
                   sdbot_ptr, theta_ptr, cellsize_ptr, level_ptr, seed_ptr, field_ptr, squash_ptr, stretchx_ptr,
                   stretchy_ptr, lognormal_ptr, fieldfromcentre_ptr)
        fields.append(np.array(field))
        seed = seed - 1
    return fields