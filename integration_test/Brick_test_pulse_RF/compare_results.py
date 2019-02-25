import matplotlib.pylab as plt


def read_pickle(file):
    import pickle
    # read pickle file
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


# numerical solution
data = read_pickle("./results/data.pickle")

# analytical solution


# make plot
fig = plt.figure(1, figsize=(6, 5))
plt.axes().set_position([0.15, 0.1, 0.8, 0.8])
# plot numerical
plt.plot(data["time"], data["displacement"]["2989"]["x"], label="x-dir")
plt.plot(data["time"], data["displacement"]["2989"]["y"], label="y-dir")
plt.plot(data["time"], data["displacement"]["2989"]["z"], label="z-dir")

plt.grid()
plt.xlim(0, 0.15)
plt.xlabel("Time [s]")
plt.ylabel("Displacement [m]")
plt.legend(loc=1)
plt.title("Node 1644")
plt.savefig("./displacement.pdf")
plt.close()
