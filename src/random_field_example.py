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
        print(seed)
    return fields


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 2           # number of realisations in one set
    max_lvl = 7     # number of levels of subdivision (2**max_lvl) is size.
    cellsize = 0.5  # Cell size
    theta = 32.0
    xcells = 64  # Number of cells x dir
    ycells = 64  # Number of cells x dir
    zcells = 64  # Number of cells x dir
    seed = -26021981
    mean = 10.0
    sd = 0.5
    lognormal = True
    fieldfromcentre = False
    # anisox = 1
    # anisoy = 1

    # everytime you call this you need a new seed number.
    fields = rand3d(n, max_lvl, cellsize, theta, xcells, ycells, zcells, seed, mean, sd, lognormal, fieldfromcentre)
    plt.imshow(fields[0][:, :, 15])
    plt.show()


