import matplotlib.pylab as plt
import numpy as np
import os


def calc_ana_sol(L, K, rho, p0, nb_ele, cycles=10, terms=100):
    r"""
    Analytical solution for the wave propagation on an elastic solid of finite length, when subjected to an pressure.

    Calculation of the solution of the wave equation is based on the analytical solution:
    Churchill, R.V. Operational Mathematics. 3rd edition. McGraw-Hill Book Company. 1972. pp: 253-257.

    :param L: Length of the solid column.
    :type L: float
    :param K: Bulk modulus solid.
    :type K: float
    :param rho: Density solid.
    :type rho: float
    :param p0: Initial pressure on the solid boundary.
    :type p0: float
    :param nb_ele: Number of elements.
    :type nb_ele: float
    :param cycles: Number of cycles of the wave traveling wave.
    :type cycles: float
    :param terms: Number of terms of the Fourier series.
    :type terms: float

    :return: time
    :rtype: np.ndarray
    :return: u - displacement of the solid
    :rtype: np.ndarray
    :return: p - pressure of the solid
    :rtype: np.ndarray
    :return: v - velocity of the solid
    :rtype: np.ndarray
    """

    # wave speed
    c = np.sqrt(K / rho)
    # solid discretisation
    H_discre = np.linspace(0, L, nb_ele + 1)
    # time discretisation
    time = np.linspace(0, (cycles * L / c), int(np.ceil(c / L) * 10))

    # variable initialisation: u = displacement; p = pressure
    u = np.zeros((H_discre.shape[0], time.shape[0]))
    v = np.zeros((H_discre.shape[0], time.shape[0]))
    p = np.zeros((H_discre.shape[0], time.shape[0]))

    for id_t, t in enumerate(time):
        for id_x, x, in enumerate(H_discre):
            summation = 0
            summation_p = 0
            summation_v = 0
            for k in range(1, terms):
                # Fourier terms
                lambda_k = (2 * k - 1) * np.pi / (2 * L)
                summation += (-1)**k / (2 * k - 1)**2 * np.sin(lambda_k * x) * np.cos(lambda_k * c * t)
                summation_p += (-1)**k * lambda_k / (2 * k - 1)**2 * np.cos(lambda_k * x) * np.cos(lambda_k * c * t)
                summation_v -= (-1)**k * lambda_k * c / (2 * k - 1)**2 * np.sin(lambda_k * x) * np.sin(lambda_k * c * t)

            u[id_x, id_t] = p0 / K * (x + 8 * L / np.pi**2 * summation)
            p[id_x, id_t] = p0 / K * (1 + 8 * L / np.pi**2 * summation_p) * K
            v[id_x, id_t] = p0 / K * (8 * L / np.pi**2 * summation_v)

    # # plots
    # plot_plot([time * c / L, time * c / L, time * c / L], [u[10, :] / (p0 / K), u[5, :] / (p0 / K), u[0, :] / (p0 / K)],
    #           ['0 H', '0.5 H', '1 H'],
    #           'Normalised time: $ct/L$ [-]', 'Normalised displacement: $u/u_{0}$ [-]', '', './', 'dis', [0, 8])
    # plot_plot([time * c / L, time * c / L, time * c / L], [v[10, :] / (p0 / K * c), v[5, :] / (p0 / K * c), v[0, :] / (p0 / K * c)],
    #           ['0 H', '0.5 H', '1 H'],
    #           'Normalised time: $ct/L$ [-]', 'Normalised velocity: $v K / p_{0} / c$ [-]', '', './', 'vel', [0, 8])
    # plot_plot([time * c / L, time * c / L, time * c / L], [p[10, :] / -p0, p[5, :] / -p0, p[0, :] / -p0],
    #           ['0 H', '0.5 H', '1 H'],
    #           'Normalised time: $ct/L$ [-]', 'Normalised pressure: $p/p_{0}$ [-]', '', './', 'press', [0, 8])
    return time, u, p, v

    
def plot_plot(x_values, y_values, label_name, x_axis_label, y_axis_label, title, path_res, save_name, xlim, y_lim=None):

    plt.figure(num=1, figsize=(9, 6), dpi=80)
    plt.axes().set_position([0.15, 0.1, 0.8, 0.8])
    # plt.gca().yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    if not label_name:
        label_name = [''] * len(y_values)

    # define colour and marker
    clr = ['b', 'r', 'k', 'g', '0.75', 'y', 'k']
    mrk = ['x', 'o', '^', 's', '+', 'h', '']
    line_style = ['-', '-', '-', '-', '-', '-', '-']
    mrk_size = [4, 3, 3, 3, 3, 3, 3]
    liwidth = [1, 1.25, 1, 1, 1, 1, 1]

    # plot for each y_value
    for i, y_data in enumerate(y_values):
        plt.plot(x_values[i], y_data, label=str(label_name[i]), color=clr[i], linewidth=liwidth[i])

    plt.xlim(xlim)
    if y_lim is not None:
        plt.ylim(y_lim)
    plt.grid()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)

    # legend location. default is loc=0 (best fit).
    plt.legend(loc=4)
    # save the figure
    plt.savefig(os.path.join(path_res, save_name) + '.png', format='png')
    plt.savefig(os.path.join(path_res, save_name) + '.eps', format='eps')
    plt.close()
    return
    
    
if __name__ == '__main__':
    calc_ana_sol(1, 20e6, 1000, -10e3, 10, cycles=10, terms=1000)

