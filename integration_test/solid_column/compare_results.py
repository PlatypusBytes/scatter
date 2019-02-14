import analytical_wave_prop as awp
import matplotlib.pylab as plt


def read_pickle(file):
    import pickle
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


def compute_analytical(E, rho, v, p, n_ele, L):
    K = E * (1 - v) / ((1 + v) * (1 - 2 * v))
    time, u, p, v = awp.calc_ana_sol(L, K, rho, -p, n_ele, cycles=16, terms=50)
    return time, u, p, v


# numerical solution
data = read_pickle("./results/data.pickle")

# analytical solution
tim, dis, stress, vel = compute_analytical(30e6, 1500, 0.2, 1000*4, 10, 10)

# make plot
fig = plt.figure(1, figsize=(6, 5))
plt.axes().set_position([0.15, 0.1, 0.8, 0.8])
# plot numerical
plt.plot(data["time"], data["velocity"][:, 0], label="numerical")
# plot analytical
plt.plot(tim, vel[10, :], label="analytical")
plt.grid()
plt.xlim(0, 1)
plt.xlabel("Time [s]")
plt.ylabel("Velocity [m/s]")
plt.legend(loc=1)
plt.savefig("./velocity.pdf")
plt.close()
