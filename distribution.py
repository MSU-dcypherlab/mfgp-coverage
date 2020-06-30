"""
distribution.py

Generates 3-dimensional (x, y, z=f(x,y)) distributions normalized to the unit cube [0, 1]^3 used by learning-coverage
algorithms in multiagent settings. Displays intermediate results of generated distributions using
matplotlib and saves (x, y, z=f(x,y)) triples to a CSV file

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/11/2020
"""

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

""" Minimum value possible when normalizing function values (to avoid div-by-0 errors) """
epsilon = 0.01


def normalize(y, use_epsilon=True):
    """
    Normalize the numpy array y between 0 + epsilon and 1. Epsilon lower bound prevents divide by zero errors later
    in coverage algorithms (points with zero weight lead to numerical errors when computing cell centroids).

    :param y: [nx1 numpy array] of values to be normalized
    :param use_epsilon: [boolean] if True, add epsilon to minimum to prevent later divby0 errors
    :return: [nx1 numpy array] normalized with values in [epsilon, 1]
    """
    if use_epsilon:
        y = y - np.amin(y) + epsilon
    else:
        y = y - np.amin(y)

    y = y / np.amax(y)
    return y


def exponential(x_star, lenscale, positive_centers, negative_centers):
    """
    Generate a 3-dimensional (x, y, z=f(x,y)) distribution by summing exponential bump functions parameterized by
    lenscale to positive_centers and subtracting from negative_centers. Evaluate this function at all points in x_star.
    Each exponential bump function is given by
        f(x, x') = ± exp( -dist(x, x')^2 / lenscale )
    where x is a 2-dimensional point in the unit square, x' is a base point of positive_centers or negative_centers, and
    the sign is determined by whether the point is in positive_centers or negative_centers

    :param x_star: [nx2 numpy array] of (x, y) pairs at which to evaluate distribution function
    :param lenscale: [scalar] lengthscale parameter of bump function: greater values correspond to "smoother" functions
    :param positive_centers: [ix2 numpy array] of points at which a positive bump function is centered
    :param negative_centers: [jx2 numpy array] of points at which a negative bump function is centered
    :return: [nx1 numpy array] containing z=f(x,y) where (x,y) are specified by points in x_star and f is
             parameterized by positive_centers, negative_centers and lenscale
    """
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


def diag():
    """
    Generate a multi-fidelity 3-dimensional (x, y, z=f(x,y)) distribution with high values on bottom-left to top-right
    diagonal. Verify correlation between two fidelities and visualize distribution with matplotlib. Pull
    random subsample of generated points and save separately for hyperparameter training. Save high and
    low fidelity distribution evaluations to CSV files.

    :return: None
    """
    out_name = "diag"

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
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    hifi_train = hifi[idx, :]
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    lofi_train = lofi[idx, :]
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
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

    # 6) visualize results
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

    # 7) save files if distribution is deemed valid
    valid = input("Save distribution?")
    if valid.lower() == "y":
        hifi_df.to_csv("Data/" + out_name + "_hifi.csv", index=False)
        lofi_df.to_csv("Data/" + out_name + "_lofi.csv", index=False)
        hifi_train_df.to_csv("Data/" + out_name + "_hifi_train.csv", index=False)
        lofi_train_df.to_csv("Data/" + out_name + "_lofi_train.csv", index=False)
        sifi_train_df.to_csv("Data/" + out_name + "_sifi_train.csv", index=False)
        prior_df.to_csv("Data/" + out_name + "_prior.csv", index=False)

    print("Done.")


def two_corners():
    """
    Generate a multi-fidelity 3-dimensional (x, y, z=f(x,y)) distribution with high values on bottom-left and top-right
    corners. Verify correlation between two fidelities and visualize distribution with matplotlib. Pull
    random subsample of generated points and save separately for hyperparameter training. Save high and
    low fidelity distribution evaluations to CSV files.

    :return: None
    """
    out_name = "anti_two_corners"

    # 1) generate grid
    delta = 0.02
    grid = np.arange(0, 1 + delta, delta)
    x_star = np.array([[i, j] for i in grid for j in grid])

    # 2) generate functions evaluated on grid and validate correlation
    y_H = exponential(x_star, lenscale=0.05,
                      positive_centers=[[0.1, 0.1], [0.9, 0.9]],
                      negative_centers=None)
    y_L = exponential(x_star, lenscale=0.3,
                      positive_centers=[[0.1, 0.9], [0.9, 0.1]],
                      negative_centers=None)
    print("Correlation: " + str(np.corrcoef(y_L, y_H)))
    hifi = np.column_stack((x_star, y_H))
    lofi = np.column_stack((x_star, y_L))
    sifi = np.vstack((hifi, lofi))

    # 3) construct sample arrays for hyperparameter training
    sample_size = int(0.1 * x_star.shape[0])
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    hifi_train = hifi[idx, :]
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    lofi_train = lofi[idx, :]
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
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

    # 6) visualize results
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

    # 7) save files if distribution is deemed valid
    valid = input("Save distribution?")
    if valid.lower() == "y":
        hifi_df.to_csv("Data/" + out_name + "_hifi.csv", index=False)
        lofi_df.to_csv("Data/" + out_name + "_lofi.csv", index=False)
        hifi_train_df.to_csv("Data/" + out_name + "_hifi_train.csv", index=False)
        lofi_train_df.to_csv("Data/" + out_name + "_lofi_train.csv", index=False)
        sifi_train_df.to_csv("Data/" + out_name + "_sifi_train.csv", index=False)
        prior_df.to_csv("Data/" + out_name + "_prior.csv", index=False)

    print("Done.")


def australian_wildfires():
    """
    Generate a multi-fidelity 3-dimensional (x, y, z=f(x,y)) distribution based on locational occurrence
    data of Australian Wildfires from Kaggle. Verify correlation and visualize distribution with matplotlib.
    Pull random subsample of generated points and save separately for hyperparameter training. Save high and
    low fidelity distribution evaluations to CSV files.

    Data retrieved from https://www.kaggle.com/carlosparadis/fires-from-space-australia-and-new-zeland May 28, 2020.
    Referenced https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html June 23, 2020.

    :return: None
    """
    # out_name = "australia5"
    hifi_sigma_n = 0.1      # std. dev. of hifi noise
    lofi_sigma_n = 0.25     # std. dev. of lofi noise

    # 1) read in raw CSV from Kaggle and filter to single date
    fires = pd.read_csv("Kaggle/AustralianWildfires/fire_archive_M6_96619.csv")
    date = "2019-08-01"
    fires = fires[fires.acq_date == date]
    plt.scatter(fires.longitude, fires.latitude)
    plt.title(f"Filtered by Date: {date}")
    plt.show()

    # 2) select only data from within boxed latitude
    fires = fires[(fires.latitude > -35) & (fires.latitude < -29) &
                  (fires.longitude > 115) & (fires.longitude < 125)]

    # 3) take X points to be (x1 = longitude, x2 = latitude) and normalize into unit square
    data = np.array([fires.longitude, fires.latitude]).T
    plt.scatter(data[:, 0], data[:, 1])
    plt.title(f"Filtered by Date: {date} and Lat/Lon")
    plt.show()

    data[:, 0] = normalize(data[:, 0], use_epsilon=False)
    data[:, 1] = normalize(data[:, 1], use_epsilon=False)

    # 4) derive KDE estimate of occurrences for density function (scipy takes each row as a dimension, not column)
    hifi_kde = scipy.stats.gaussian_kde(data.T)

    # 5) create low-fidelity KDE in which bandwidth/lengthscale is longer
    lofi_kde = copy.deepcopy(hifi_kde)
    lofi_kde.set_bandwidth(hifi_kde.factor * 4)

    # 6) use KDE models to predict on grid and normalize
    delta = 0.02
    grid = np.arange(0, 1 + delta, delta)
    x_star = np.array([[i, j] for i in grid for j in grid])


    y_H = hifi_kde(x_star.T)        # scipy takes rows as dimensions here
    y_L = lofi_kde(x_star.T)        # scipy takes rows as dimensions here

    y_H = normalize(y_H)
    y_L = normalize(y_L)

    # 7) verify between-fidelity correlation
    print("Correlation: " + str(np.corrcoef(y_L, y_H)))
    hifi = np.column_stack((x_star, y_H))
    lofi = np.column_stack((x_star, y_L))

    # 8) construct sample arrays WITH ADDED SAMPLE NOISE for hyperparameter training
    sample_size = int(0.1 * x_star.shape[0])

    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    hifi_train = hifi[idx, :]
    hifi_train[:, 2] += np.random.default_rng().normal(loc=0, scale=hifi_sigma_n, size=sample_size)
    idx = np.random.randint(0, x_star.shape[0], size=sample_size)
    lofi_train = lofi[idx, :]
    lofi_train[:, 2] += np.random.default_rng().normal(loc=0, scale=lofi_sigma_n, size=sample_size)
    # sifi_train = np.vstack((hifi_train, lofi_train))

    # 9) construct small prior array for initial conditioning in simulation WITH ADDED SAMPLE NOISE
    sample_delta = 0.2
    sample_grid = np.arange(0, 1 + sample_delta, sample_delta)
    x_prior = np.array([[i, j] for i in sample_grid for j in sample_grid])
    prior_idx = [np.logical_and(np.isclose(x[0], x_star[:, 0]), np.isclose(x[1], x_star[:, 1])) for x in x_prior]
    y_prior = np.array([y_L[idx] for idx in prior_idx])
    y_prior += np.random.default_rng().normal(loc=0, scale=lofi_sigma_n, size=y_prior.shape)
    prior = np.column_stack((x_prior, y_prior))

    # 10) construct dataframes from arrays
    hifi_df = pd.DataFrame(hifi)
    hifi_df.columns = ["X", "Y", "f_H"]
    lofi_df = pd.DataFrame(lofi)
    lofi_df.columns = ["X", "Y", "f_L"]
    hifi_train_df = pd.DataFrame(hifi_train)
    hifi_train_df.columns = ["X", "Y", "f_H_train"]
    lofi_train_df = pd.DataFrame(lofi_train)
    lofi_train_df.columns = ["X", "Y", "f_L_train"]
    # sifi_train_df = pd.DataFrame(sifi_train)
    # sifi_train_df.columns = ["X", "Y", "f_S_train"]
    prior_df = pd.DataFrame(prior)
    prior_df.columns = ["X", "Y", "f_prior"]

    # 11) visualize results
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

    # 12) save files if distribution is deemed valid
    valid = input("Save distribution?")
    if valid.lower() == "y":
        hifi_df.to_csv(f"Data/{out_name}_hifi.csv", index=False)
        lofi_df.to_csv(f"Data/{out_name}_lofi.csv", index=False)
        hifi_train_df.to_csv(f"Data/{out_name}_hifi_train.csv", index=False)
        lofi_train_df.to_csv(f"Data/{out_name}_lofi_train.csv", index=False)
        # sifi_train_df.to_csv(f"Data/{out_name}_sifi_train.csv", index=False)
        prior_df.to_csv(f"Data/{out_name}_prior.csv", index=False)
    fig.savefig(f"Images/{out_name}_distribution.png")

    print("Done.")


if __name__ == "__main__":
    """
    Run selected distribution-generating function.
    """
    australian_wildfires()