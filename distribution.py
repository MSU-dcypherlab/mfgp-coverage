import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(y):
    y = y - np.amin(y)
    y = y / np.amax(y)
    return y

def exponential(x_star, lenscale, positive_centers, negative_centers):
    y = np.zeros(x_star.shape[0])

    if positive_centers:
        positive_centers = np.array(positive_centers)
        for center in positive_centers:
            distances = np.sum((x_star[:, [0, 1]] - center)**2, axis=1)  # dist_from_center = dx^2 + dy^2 for each row
            kernel = np.exp(-distances / lenscale)
            y += kernel
    if negative_centers:
        negative_centers = np.array(negative_centers)
        for center in negative_centers:
            distances = np.sum((x_star[:, [0, 1]] - center)**2, axis=1)  # dist_from_center = dx^2 + dy^2 for each row
            kernel = np.exp(-distances / lenscale)
            y -= kernel

    return normalize(y)

if __name__ == "__main__":
    # 1) generate grid
    delta = 0.02
    grid = np.arange(0, 1 + delta, delta)
    x_star = np.array([[i, j] for i in grid for j in grid])

    # 2) generate functions evaluated on grid and validate correlation
    y_H = exponential(x_star, lenscale=0.1,
                      positive_centers=[[0.1, 0.1], [0.9, 0.9]],
                      negative_centers=[[0.1, 0.9], [0.9, 0.1]])
    y_L = exponential(x_star, lenscale=0.2,
                      positive_centers=[[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]],
                      negative_centers=None)
    print("Correlation: " + str(np.corrcoef(y_L, y_H)))
    hifi = np.column_stack((x_star, y_H))
    lofi = np.column_stack((x_star, y_L))
    sifi = np.vstack((hifi, lofi))

    # 3) construct sample arrays for hyperparameter training
    sample_size = int(0.1 * x_star.shape[0])
    idx = np.random.randint(0, x_star.shape[0],  size=sample_size)
    hifi_train = hifi[idx, :]
    idx = np.random.randint(0, x_star.shape[0],  size=sample_size)
    lofi_train = lofi[idx, :]
    idx = np.random.randint(0, x_star.shape[0],  size=sample_size)
    sifi_train = sifi[idx, :]

    # 4) construct small prior array for initial conditioning in simulation
    prior_points = [0.16, 0.5, 0.84]
    x_prior = np.array([[i, j] for i in prior_points for j in prior_points])
    prior_idx = [np.logical_and(x[0] == x_star[:, 0], x[1] == x_star[:, 1]) for x in x_prior]
    y_prior = [y_L[idx] for idx in prior_idx]
    prior = np.column_stack((x_prior, y_prior))

    # 5) construct dataframes from arrays
    hifi_df = pd.DataFrame(hifi)
    hifi_df.columns = ["X", "Y", "f_H"]
    lofi_df = pd.DataFrame(lofi)
    lofi_df.columns = ["X", "Y", "f_L"]
    hifi_train_df = pd.DataFrame(hifi_train)
    hifi_train_df.columns = ["X", "Y", "f_H_train"]
    lofi_train_df = pd.DataFrame(lofi_train)
    lofi_train_df.columns = ["X", "Y", "f_L_train"]
    sifi_train_df = pd.DataFrame(sifi_train)
    sifi_train_df.columns = ["X", "Y", "f_S_train"]
    prior_df = pd.DataFrame(prior)
    prior_df.columns = ["X", "Y", "f_prior"]

    # 5) save files
    f_name = "diag"
    hifi_df.to_csv("Data/" + f_name + "_hifi.csv")
    lofi_df.to_csv("Data/" + f_name + "_lofi.csv")
    hifi_train_df.to_csv("Data/" + f_name + "_hifi_train.csv")
    lofi_train_df.to_csv("Data/" + f_name + "_lofi_train.csv")
    sifi_train_df.to_csv("Data/" + f_name + "_sifi_train.csv")
    prior_df.to_csv("Data/" + f_name + "_prior.csv")

    # 9) visualize results
    fig = plt.figure()

    ax = fig.add_subplot(321)
    ax.scatter(x_star[:, 0], x_star[:, 1], c=y_H[:])
    ax.set_aspect("equal")
    ax.set_title("High Fidelity")

    ax = fig.add_subplot(323)
    ax.scatter(x_star[:, 0], x_star[:, 1], c=y_L[:])
    ax.set_aspect("equal")
    ax.set_title("Low Fidelity")

    diff = y_H - y_L
    ax = fig.add_subplot(325)
    ax.scatter(x_star[:, 0], x_star[:, 1], c=diff[:])
    ax.set_aspect("equal")
    ax.set_title("Difference")

    ax = fig.add_subplot(322)
    ax.scatter(hifi_train[:, 0], hifi_train[:, 1], c=hifi_train[:, 2])
    ax.set_aspect("equal")
    ax.set_title("Hifi Train")

    ax = fig.add_subplot(324)
    ax.scatter(lofi_train[:, 0], lofi_train[:, 1], c=lofi_train[:, 2])
    ax.set_aspect("equal")
    ax.set_title("Lofi Train")

    ax = fig.add_subplot(326)
    ax.scatter(prior[:, 0], prior[:, 1], c=prior[:, 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title("Prior Points")

    plt.show()
    print("Done.")

