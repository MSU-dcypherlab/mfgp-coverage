"""
simulator.py

Simulates learning-coverage algorithms on a given density function over the unit square.
Given no knowledge or incomplete knowledge of the density function beforehand, agents must balance
exploration (learning the density function) with exploitation (covering the density function) to converge
upon a locally optimal solution of the density-respective coverage problem.

We refer readers to Todescato et. al. "Multi-robots Gaussian estimation and coverage..." for an introduction
to the dual problem at hand. We model the unknown function as a (single- or multi-fidelity) Gaussian Process.

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/11/2020
"""

import sys
import copy
import random
import cProfile
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib import path
from gaussian_process import MFGP, SFGP
from plotter import Plotter
import six
sys.modules['sklearn.externals.six'] = six  # Workaround to import MLRose https://stackoverflow.com/a/62354885
import mlrose


""" Boundary cushion to be used in computations with and plots of unit square """
eps = 0.1

""" Delimiter used in console output """
line_break = "\n" + "".join(["*" for i in range(100)]) + "\n"


#######################################################################################################################
# Helper functions
#######################################################################################################################


def init_MFGP(hyp, prior):
    """
    Initialize the MFGP model to be utilized in this simulation with or without a prior to condition upon.
    If provided, the prior is assumed to be low-fidelity.
    To obtain hyperparameters, the MFGP model must be trained on a given dataset before simulation. See trainer.py.

    :param hyp: [1x9 pandas DF] of log-scaled hyperparameters to use in model (9 is number of hyperparameters)
                log-scaled hyp take the form [mu_lo, s^2_lo, L_lo, mu_hi, s^2_hi, L_hi, rho, noise_lo, noise_hi]
    :param prior: [nx3 pandas DF] of (x,y,z=f(x,y)) triples to condition model upon
                  if empty, model will not be conditioned on any prior
    :return: [MFGP object] initialized MFGP model to be used in simulation
    """
    if prior is not None and len(prior) > 0:
        p = np.vstack(prior.values.tolist())
        X_L = np.reshape(p[:, [0, 1]], (-1,2))  # all rows, first two columns are X,Y of lofi prior
        y_L = np.reshape(p[:, 2], (-1,1))       # all rows, third column is F of lofi prior
    else:
        X_L = np.empty([0, 2])
        y_L = np.empty([0, 1])
    X_H = np.empty([0, 2])
    y_H = np.empty([0, 1])
    len_L = 1
    len_H = 1
    model = MFGP(X_L, y_L, X_H, y_H, len_L, len_H)

    h_arr = np.array(hyp.values.tolist()[0])  # convert hyperparameters from dataframe to list
    model.hyp = h_arr

    return model


def init_SFGP(hyp, prior):
    """
    Initialize the SFGP model to be utilized in this simulation with or without a prior to condition upon.
    To obtain hyperparameters, the SFGP model must be trained on a given dataset before simulation. See trainer.py.

    :param hyp: [1x4 pandas DF] of log-scaled hyperparameters to use in model (4 is number of hyperparameters)
                log-scaled hyp take the form [mu, s^2, L, noise]
    :param prior: [nx3 pandas DF] of (x,y,z=f(x,y)) triples to condition model upon
                  if empty, model will not be conditioned on any prior
    :return: [SFGP object] initialized SFGP model to be used in simulation
    """
    if prior is not None and len(prior) > 0:
        p = np.vstack(prior.values.tolist())
        X = np.reshape(p[:, [0, 1]], (-1, 2))   # all rows, first two columns are X,Y of lofi prior
        y = np.reshape(p[:, 2], (-1, 1))        # all rows, third column is F of lofi prior
    else:
        X = np.empty([0, 2])
        y = np.empty([0, 1])
    len_sf = 1
    model = SFGP(X, y, len_sf)

    h_arr = np.array(hyp.values.tolist()[0])  # convert hyperparameters from dataframe to list
    model.hyp = h_arr

    return model


def in_polygon(xq, yq, xv, yv):
    """
    Determines if query points (xq, yq) are contained in the polygon specified by (xv, yv)
    Translates Matlab inpolygon implementation described here: https://www.mathworks.com/help/matlab/ref/inpolygon.html
    Referenced https://stackoverflow.com/a/49733403 May 23, 2020

    :param xq: [nx1 numpy array] of query point x-coordinates
    :param yq: [nx1 numpy array] of query point y-coordinates
    :param xv: [nx1 numpy array] of bounding polygon x-coordinates
    :param yv: [nx1 numpy array] of bounding polygon y-coordinates
    :return: [nx1 numpy array] of boolean values indicating if i-th (xq, yq) point is inside (xv, yv) polygon
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)


def poly_area(x, y):
    """
    Implementation of the Shoelace formula to compute 2D polygonal area (https://en.wikipedia.org/wiki/Shoelace_formula)
    Referenced https://stackoverflow.com/a/30408825 May 24, 2020

    :param x: [nx1 numpy array] of x points defining polygon
    :param y: [nx1 numpy array] of y points defining polygon
    :return: [scalar] area of polygon specified by (x,y) points
    """
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def in_box(points, bounding_box):
    """
    Determine if a given set of 2D points is inside a given 2D bounding box
    Referenced https://stackoverflow.com/a/33602171 May 23, 2020

    :param points: [nx2 numpy array] of (x,y) points to be checked
    :param bounding_box: [1x4 numpy array] containing limits [x_min, x_max, y_min, y_max]
    :return: [nx1 numpy array] of boolean values indicating if the i-th (x,y) point is inside the bounding box
    """
    return np.logical_and(np.logical_and(bounding_box[0] - eps <= points[:, 0],
                                         points[:, 0] <= bounding_box[1] + eps),
                          np.logical_and(bounding_box[2] - eps <= points[:, 1],
                                         points[:, 1] <= bounding_box[3] + eps))


def voronoi_bounded(points, bounding_box):
    """
    Compute 2D Voronoi partition bounded within a bounding_box
    Referenced https://stackoverflow.com/a/33602171 May 23, 2020

    :param points: [nx2 numpy array] of seed points around which to construct Voronoi diagram
    :param bounding_box: [1x4 numpy array] containing limits [x_min, x_max, y_min, y_max]
    :return: [scipy Voronoi object] with filtered_points and filtered_regions fields corresponding to the points
             and regions inside the bounding_box
    """
    # Select points inside the bounding box
    i = in_box(points, bounding_box)
    # Mirror points left, right, down, and up around bounding box to artificially bound Voronoi diagram
    points_center = points[i, :]
    points_left = np.copy(points_center)
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0] + eps)
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0] + eps)
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2] + eps)
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1] + eps)
    points = np.append(points_center,
                       np.append(np.append(points_left,
                                           points_right,
                                           axis=0),
                                 np.append(points_down,
                                           points_up,
                                           axis=0),
                                 axis=0),
                       axis=0)
    # Compute Voronoi using scipy
    vor = Voronoi(points)
    # Save points and regions within bounds as "filtered"
    vor.filtered_points = points_center     # center points are the original points
    vor.filtered_regions = \
        np.array(vor.regions)[vor.point_region[:vor.npoints//5]]     # first 1/5 of regions correspond to center
    return vor


def compute_loss(vor, truth_arr):
    """
    Compute the loss of a given agent configuration according to loss function specified in
    Equation 2 of Todescato et. al. "Multi-robots Gaussian estimation and coverage..."

    :param vor: [scipy Voronoi object] bounded Voronoi partition seeded by current agent configuration,
                with bounded points and regions specified in filtered_points and filtered_regions fields
    :param truth_arr: [nx3 numpy array] of (x,y,z) triples where z=f(x,y) is the ground truth function at each point
    :return: [scalar] loss of current agent configuration
    """
    loss = 0

    # 1) iterate over each cell in voronoi partition and compute loss
    for i, cell in enumerate(vor.filtered_regions):
        # 2) select only the truth points in this cell from truth_arr
        vertices = vor.vertices[cell, :]
        in_indices = in_polygon(truth_arr[:, 0], truth_arr[:, 1], vertices[:, 0], vertices[:, 1])
        in_points = truth_arr[in_indices, :]

        # 3) compute cell loss by Equation 2 in Todescato et. al.
        center = vor.filtered_points[i, :]
        distances = np.sum((in_points[:, [0, 1]] - center)**2, axis=1)  # dist_from_center = dx^2 + dy^2 for each row
        point_loss = distances * in_points[:, 2]        # loss_of_each_point = dist * f_val
        cell_area = poly_area(vertices[:, 0], vertices[:, 1])
        cell_loss = np.mean(point_loss) * cell_area     # cell_loss = integral(point_loss), discretized as average
        loss += cell_loss

        # plt.plot(in_points[:,0], in_points[:,1], 'ro')        # debug loss computation of each cell
        # plt.plot(center[0], center[1], 'go')
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    # 4) return loss summed over all cells
    return loss


def compute_centroids(vor, x_star, mu_star):
    """
    Compute the weighted centroids of a Voronoi partition over 2D domain x_star using the predicted mean mu_star
    output by the GP model (more generally, mu_star is the weighting function)
    When Voronoi partitions are iteratively seeded by the previous step's centroids, this implements Lloyd's Algorithm

    :param vor: [scipy Voronoi object] bounded Voronoi partition seeded by
                current agent configuration OR previous step's centroids (to implement Lloyd),
                with bounded points and regions specified in filtered_points and filtered_regions fields
    :param x_star: [nx2 numpy array] of (x,y) pairs at which the predicted mean (mu_star) is output by the GP model
    :param mu_star: [nx1 numpy array] of mu(x,y) estimates of posterior mean output by the GP model
                    more generally, the weighting function to use on points in x_star when computing weighted centroids
    :return: [nAgentsx2 numpy array] of weighted centroids (x,y) of each cell
    """
    centroids = np.empty([0, 2])

    # 1) iterate over each cell in bounded voronoi partition and compute centroid
    for i, cell in enumerate(vor.filtered_regions):
        # 2) select only the truth points INSIDE this cell from truth_arr
        vertices = vor.vertices[cell, :]
        in_indices = in_polygon(x_star[:, 0], x_star[:, 1], vertices[:, 0], vertices[:, 1])
        in_points = x_star[in_indices, :]
        in_means = mu_star[in_indices, :]

        # 3) compute weighted cell centroid by Equation 1 in Todescato et. al.
        cell_area = poly_area(vertices[:, 0], vertices[:, 1])
        f_integral = np.mean(in_means[:, 0]) * cell_area   # discretized scalar integral of f over cell
        weights = np.column_stack((in_means[:, 0], in_means[:, 0]))     # create nx2 array from nx1 array of f_vals
        weighted_points = np.multiply(weights, in_points[:, [0, 1]])   # (x,y) weighted by f elementwise
        weighted_integral = np.mean(weighted_points, axis=0) * cell_area    # discretized 1x2 integral of (x*f, y*f)
        centroid = weighted_integral / f_integral   # mean 1x2 weighted location of cell, i.e., centroid

        # 4) sanity check: if centroid is out of domain, snap it back inside domain
        if centroid[0] < np.amin(x_star[:, 0]):     # check x lower bound
            centroid[0] = np.amin(x_star[:, 0])
        if centroid[0] > np.amax(x_star[:, 0]):     # check x upper bound
            centroid[0] = np.amax(x_star[:, 0])
        if centroid[1] < np.amin(x_star[:, 1]):     # check y lower bound
            centroid[1] = np.amin(x_star[:, 1])
        if centroid[1] > np.amax(x_star[:, 1]):     # check y upper bound
            centroid[1] = np.amax(x_star[:, 1])

        # 5) add to collection of centroids and continue
        centroids = np.vstack((centroids, centroid))

        # plt.figure()                              # debug computed centroid for each cell
        # plt.scatter(in_points[:, 0], in_points[:, 1], c=in_means[:, 0])
        # plt.plot(centroid[0], centroid[1], 'k+')
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    return centroids


def compute_max_var(vor, truth_arr, var_star):
    """
    Find points of maximum posterior variance output by the GP model in each cell of a given Voronoi partition.

    :param vor: [scipy Voronoi object] bounded Voronoi partition seeded by
                current agent configuration OR previous step's centroids (if implementing Lloyd),
                with bounded points and regions specified in filtered_points and filtered_regions fields
    :param truth_arr: [nx3 numpy array] of (x,y,z) triples where z=f(x,y) is the ground truth function at each point
    :param var_star: [nxn numpy array] of cov(x,x') estimates of posterior variance (diagonal contains variances)
    :return: [2 value tuple] of
        [nAgentsx2 numpy array] of (x,y) points maximizing variance in each cell (argmax(var) points of each cell)
        [nAgentsx1 numpy array] of maximum variance in each cell (variance evaluated at argmax in each cell)
    """
    argmax_var_t = np.empty([0, 2])
    max_var_t = np.empty([0, 1])
    var = np.diag(var_star)     # point variances are diagonal of cov matrix

    # 1) iterate over each cell in voronoi partition and find max, argmax
    for i, cell in enumerate(vor.filtered_regions):
        # 2) select only the points in this cell from truth_arr
        vertices = vor.vertices[cell, :]
        in_indices = in_polygon(truth_arr[:, 0], truth_arr[:, 1], vertices[:, 0], vertices[:, 1])
        in_points = truth_arr[in_indices, :]
        in_var = var[in_indices]

        # 3) find max, argmax of var in this cell and save
        max_var = np.amax(in_var)
        argmax_var = in_points[np.argmax(in_var), [0, 1]]    # take (x,y) only
        argmax_var_t = np.vstack((argmax_var_t, argmax_var))
        max_var_t = np.vstack((max_var_t, max_var))

        # plt.scatter(in_points[:, 0], in_points[:, 1], c=in_var)       # debug max var points of each cell
        # plt.plot(argmax_var[0], argmax_var[1], 'k+')
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    return argmax_var_t, max_var_t


def compute_sample_points(model, x_star, threshold, console):
    """
    Given GP model, determine points to sample in order to reduce maximum variance below a given threshold.
    Choose points most efficiently by sampling point with maximal predictive variance on each iteration.
    Utilized in Choi doubling algorithms.

    :param model: [MFGP or SFGP object] GP model with valid hyperparameters and given observations
    :param x_star: [nx2 numpy array] of (x,y) pairs at which the predicted mean/var is output by the GP model
    :param threshold: [scalar] value below which maximum variance needs to be reduced
    :param console: [boolean] indicates if computational progress should be displayed on console
    :return: [nx2 numpy array] of (x,y) pairs at which samples must be taken in order to most efficiently reduce
             predictive variance beneath a given threshold
    """
    temp_model = copy.deepcopy(model)   # use copy when determining sample points to leave original sample set unchanged
    mu_star, var_star = temp_model.predict(x_star)
    var = np.diag(var_star)
    max_var = np.amax(var)
    sample_points = np.empty([0, 2])    # store to-sample-points in here and return

    while max_var > threshold:

        # status update
        print("Current max var: " + str(max_var) + " // Target threshold: " + str(threshold)) if console else None
        print("Finding sample point: " + str(sample_points.shape[0] + 1)) if console else None

        # determine next point to sample
        argmax_var = x_star[np.argmax(var)]     # find point at which predictive variance is maximized
        argmax_mu = mu_star[np.argmax(var)]     # use estimated mean value as sample value

        # add point to to-be-sampled set
        x_addition = np.array(argmax_var).reshape((1, -1))
        sample_points = np.vstack((sample_points, x_addition))

        # update temp model with new sample point, taking predicted mean to be the observed value
        y_addition = np.array(argmax_mu).reshape((1, -1))
        if isinstance(temp_model, SFGP):
            temp_model.updt(x_addition, y_addition)
        elif isinstance(temp_model, MFGP):
            temp_model.updt_hifi(x_addition, y_addition)
        else:
            raise TypeError("Invalid model type: must be SFGP or MFGP")

        # recompute prediction and continue adding points to reduce variance if necessary
        mu_star, var_star = temp_model.predict(x_star)
        var = np.diag(var_star)
        max_var = np.amax(var)

    # return sample points when temp model's maximum predictive variance is below threshold
    return sample_points


def compute_sample_clusters(vor, sample_points):
    """
    Given a set of points to sample and a bounded Voronoi diagram of the current Lloyd iteration of the agents,
    determine which points fall in each Voronoi cell and assign these points to the agent owning that cell.

    :param vor: [scipy Voronoi object] bounded Voronoi partition seeded by
                current agent configuration OR previous step's centroids (if implementing Lloyd),
                with bounded points and regions specified in filtered_points and filtered_regions fields
    :param sample_points: [nx2 numpy array] of (x,y) pairs at which samples must be taken
    :return: [list of nx2 numpy arrays] where list entry i contains an nx2 numpy array of (x,y) pairs at which
             samples must be taken by agent i (i.e., contains points inside of cell i)
    """
    clusters = []

    # 1) iterate over each cell in Voronoi partition and determine the sample points inside of this cell
    for i, cell in enumerate(vor.filtered_regions):

        # 2) select only the points in this cell from sample_points
        vertices = vor.vertices[cell, :]
        in_indices = in_polygon(sample_points[:, 0], sample_points[:, 1], vertices[:, 0], vertices[:, 1])
        in_points = sample_points[in_indices, :]

        # 3) save sample points inside of this cell into list
        clusters.append(in_points)

        # plt.figure()                                        # debug sample point clusters
        # vertices = vor.vertices[cell + [cell[0]], :]
        # plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
        # plt.plot(sample_points[:, 0], sample_points[:, 1], 'k+')
        # plt.plot(in_points[:, 0], in_points[:, 1], 'r+')
        # plt.show()

    return clusters


def compute_sample_tsp(clusters):
    """
    Given a set of sample points assigned to each agent, find a near-optimal TSP tour through the sample points
    for each agent to follow.
    Ref. https://towardsdatascience.com/solving-travelling-salesperson-problems-with-python-5de7e883d847 June 14, 2020

    :param clusters: [list of nx2 numpy arrays] where list entry i contains an nx2 numpy array of (x,y) pairs at which
                     samples must be taken by agent i (i.e., i-th entry contains points to be sampled by agent i)
    :return: [list of nx2 numpy arrays] where list entry i contains an nx2 numpy array of (x,y) pairs at which
             samples must be taken by agent i, and each nx2 array is SORTED in optimal TSP order
    """
    tours = []

    # 1) iterate over each cluster of sample points
    for i, cluster in enumerate(clusters):

        tour = np.empty((0, 2))
        if cluster.shape[0] > 0:
            # 2) compute a TSP tour through this non-empty cluster using MLRose
            coords_list = [tuple(coord) for coord in cluster]               # convert numpy to list of tuples for mlrose
            problem = mlrose.TSPOpt(length=len(coords_list),
                                    coords=coords_list, maximize=False)     # define optimization problem in mlrose
            solution, fitness = mlrose.genetic_alg(problem, mutation_prob=0.2,
                                                   max_attempts=100, random_state=2)     # compute TSP tour

            # 3) rearrange cluster points according to tour and save in tours
            tour = cluster[solution]        # solution holds indices of points in optimal order
        tours.append(tour)

        # print(f"TSP {i}: Tour {tour} // Cost {fitness}")        # debug TSP tour for this cluster
        # plt.figure()
        # plt.plot(tour[:, 0], tour[:, 1], 'k-')
        # plt.plot(tour[:, 0], tour[:, 1], 'k+')
        # for j in range(len(tour)):
        #     plt.annotate(s=j, xy=(tour[j, 0], tour[j, 1]))
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    return tours


def choi_threshold(var_star):
    """
    Given current predictive variance, determine threshold below which uncertainty should be reduced in this period of
    Choi doubling algorithm.

    :param var_star: [nxn numpy array] of cov(x,x') estimates of posterior variance (diagonal contains variances)
    :return: [scalar] threshold value below which uncertainty should be reduced on this step
    """
    var = np.diag(var_star)
    max_var = np.amax(var)
    return max_var / 2


def choi_double(period):
    """
    Given current period of Choi doubling algorithm, determine the number of iterations in this period
    Begin with 10 iterations in first period, and double for all subsequent periods.

    :param period: [scalar] current period of Choi doubling algorithm
    :return: [scalar] number of iterations in this period
    """
    return 8*2**period


#######################################################################################################################
# Control Algorithms
#######################################################################################################################


def todescato(title, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log):
    """
    Implement Algorithm 1 of Todescato et. al. "Multi-robots Gaussian estimation and coverage..." using a
    single-fidelity GP model.
    Support single-fidelity and multi-fidelity models; model choice depends on hyperparameters passed in.

    :param sim_num: [scalar] index number of current simulation (relevant when running multiple simulations)
    :param iterations: [scalar] number of iterations to run simulation
    :param agents: [scalar] number of agents being simulated
    :param positions: [nAgentsx2 numpy array] of initial (x,y) points of agents
    :param truth: [nx3 pandas DF] triples of (x,y,z=f(x,y)) where z=f is the ground truth function at each point
    :param prior: [mx3 pandas DF] triples of (x,y,z~f(x,y)) where z~f is a low-fidelity prior estimate of the ground
                  truth function at each point, and m << n such that the prior estimate is given only at a few points
    :param hyp: [1x9 pandas DF] of log-scaled hyperparameters to use in MFGP model if multi-fidelity simulation
                log-scaled hyp take the form [mu_lo, s^2_lo, L_lo, mu_hi, s^2_hi, L_hi, rho, noise_lo, noise_hi]
                OR
                [1x4 pandas DF] of log-scaled hyperparameters to use in SFGP model if single-fidelity simulation
                log-scaled hyp take the form [mu, s^2, l, noise]
    :param console: boolean indicating whether to display simulation progress on console
    :param plotter: boolean indicating whether to plot simulation progress using plotter.py
    :param log: boolean indicating whether to log simulation progress into CSV for performance analysis
    :return: [3 value tuple] of
        [list of dictionaries] log of loss by iteration
        [list of dictionaries] log of agent positions and actions by iteration
        [list of dictionaries] log of samples taken by agent over the course of simulation
    """
    # 0) initialize logging lists, plotter, and determine fidelity
    loss_log, agent_log, sample_log = [], [], [] if log else None
    plotter.reset() if plotter else None
    if len(hyp.columns) == 4:
        fidelity = "S"  # use singlefidelity model
    elif len(hyp.columns) == 9:
        fidelity = "M"  # use multifidelity model
    else:
        raise TypeError("Hyperparameters must be of length 4 (single-fidelity) or 9 (multi-fidelity)")

    print(line_break + title + line_break) if console else None

    # 1) initialize model with hyperparameters and empty prior
    if fidelity == "S":
        model = init_SFGP(hyp, prior=None)
    else:
        model = init_MFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points, y truth points, loss and bounding box of domain
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [2]]  # all rows, third column is ground truth y points
    bounding_box = np.array([np.amin(x_star[:, 0]), np.amax(x_star[:, 0]),
                             np.amin(x_star[:, 1]), np.amax(x_star[:, 1])])  # [x_min, x_max, y_min, y_max]
    loss = []

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0)) if console else None

    # 4) initialize model with prior and force-update model
    if fidelity == "S":
        model = init_SFGP(hyp, prior=prior)
        model.updt_info(model.X, model.y)
    else:
        model = init_MFGP(hyp, prior=prior)
        model.updt_info(model.X_L, model.y_L, model.X_H, model.y_H)

    # 5) compute prediction given prior and initialize relevant explore/exploit decision variables
    mu_star, var_star = model.predict(x_star)
    var = np.diag(var_star)
    max_var_t = np.amax(var) * np.ones((agents, 1))
    prob_explore_t = max_var_t / max_var_0 * np.ones((agents, 1))
    explore_t = np.zeros((agents, 1))   # initialize to zero so agents do not sample on first iteration
    centroids_t = positions     # initialize centroids governing Lloyd iterations to current positions
    period = 0                  # irrelevant in this simulation, but necessary for logging consistency

    # 6) begin iterative portion of algorithm
    for iteration in range(iterations):

        print(f"\nBegin Iteration {iteration} of Simulation {sim_num} of {title}") if console else None

        # 7) record samples from each agent on explore step (Todescato "Listen")
        x_new = np.empty([0, 2])    # store new sample points
        y_new = np.empty([0, 1])    # store new samples
        id_new = np.empty([0, 1])   # store agent ids that sampled
        for i in range(agents):
            if explore_t[i] == 1:  # this robot is on an explore step: take sample
                x_sample = positions[i, :]
                sample_idx = np.logical_and(truth_arr[:, 0] == x_sample[0], truth_arr[:, 1] == x_sample[1])
                y_sample = truth_arr[sample_idx, 2]  # retrieve f_val at matching point
                print(f"Robot {i} explored {x_sample} and sampled {y_sample}") if console else None
                x_new = np.vstack((x_new, x_sample))
                y_new = np.vstack((y_new, y_sample))
                id_new = np.vstack((id_new, i))
            elif iteration > 0:     # 0th iteration is for initialization purposes only
                print(f"Robot {i} exploited to {centroids_t[i, :]}") if console else None

        # 8) update GP model and estimates (Todescato "Estimate update")
        if fidelity == "S":
            model.updt(x_new, y_new)
        else:
            model.updt_hifi(x_new, y_new)
        mu_star, var_star = model.predict(x_star)

        # 9) compute loss given current positions
        loss_vor = voronoi_bounded(positions, bounding_box)
        loss_t = compute_loss(loss_vor, truth_arr)
        loss.append(loss_t)

        # 10) update partitions and centroids given current estimate (Todescato "Partition and centroids update")
        lloyd_vor = voronoi_bounded(centroids_t, bounding_box)
        centroids_t = compute_centroids(lloyd_vor, x_star, mu_star)

        # 11) compute points of max variance
        argmax_var_t, max_var_t = compute_max_var(lloyd_vor, truth_arr, var_star)

        # 12) print to console, update log, and plot for this iteration
        # (note: period is logged in all simulations for consistency, and DOES NOT apply here)
        if console:
            print(f"Period {period}")
            print(f"Fidelity {fidelity}")
            print(f"Current loss: {loss_t}")
            print(f"Max var by cell: {max_var_t.flatten()}")
            print(f"Normalizing max var: {max_var_0}")
            print(f"Probability of exploration: {prob_explore_t.flatten()}")
            print(f"Decision of exploration: {explore_t.flatten()}")
            print(f"End Iteration {iteration}")
        if log:
            loss_log.append({"SimNum": sim_num, "Iteration": iteration, "Period": period,
                             "Fidelity": fidelity, "Loss": loss_t})
            for i in range(agents):
                agent_log.append({"SimNum": sim_num, "Iteration": iteration, "Period": period,
                                  "Fidelity": fidelity, "Agent": i,
                                  "X": positions[i, 0], "Y": positions[i, 1],
                                  "XMax": argmax_var_t[i, 0], "YMax": positions[i, 1],
                                  "VarMax": max_var_t[i, 0], "Var0": max_var_0,
                                  "XCentroid": centroids_t[i, 0], "YCentroid": centroids_t[i, 1],
                                  "ProbExplore": prob_explore_t[i, 0], "Explore": explore_t[i, 0]})
            for i in range(id_new.size):
                sample_log.append({"SimNum": sim_num, "Iteration": iteration, "Period": period, "Fidelity": fidelity,
                                   "Agent": id_new[i, 0], "X": x_new[i, 0], "Y": x_new[i, 1], "Sample": y_new[i, 0]})
        if plotter:
            plotter.plot_explore(prob_explore_t, explore_t)
            plotter.plot_mean(x_star, mu_star)
            plotter.plot_var(x_star, var_star)
            plotter.plot_loss_vor(loss_vor, truth_arr, explore_t)
            plotter.plot_loss(loss)
            plotter.plot_lloyd_vor(lloyd_vor, centroids_t, truth_arr)
            plotter.show()

        # 13) based on max variance, make next iteration's explore/exploit decision
        prob_explore_t = max_var_t / max_var_0
        explore_t = np.array([random.random() < cutoff for cutoff in prob_explore_t])  # Bernoulli wrt prob_explore_t

        # 14) update agent positions (Todescato "Target-Points transmission")
        for i in range(agents):
            if explore_t[i, 0]:
                positions[i, :] = argmax_var_t[i, :]
            else:
                positions[i, :] = centroids_t[i, :]

    # 15) return log dictionary lists to driver function, which will save them into a dataframe
    return loss_log, agent_log, sample_log


def choi(title, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log):
    """
    Implement "switching" algorithm of Choi et. al.  "Swarm intelligence for achieving the global maximum..." with
    a doubling trick inspired by Besson et. al. "What Doubling Tricks Can and Can't Do..."
    Run periods of exponentially-growing length in which agents explore to reduce below an uncertainty threshold,
    then exploit for the remainder of the period.
    Support single-fidelity and multi-fidelity models; model choice depends on hyperparameters passed in.

    :param sim_num: [scalar] index number of current simulation (relevant when running multiple simulations)
    :param iterations: [scalar] number of iterations to run simulation
    :param agents: [scalar] number of agents being simulated
    :param positions: [nAgentsx2 numpy array] of initial (x,y) points of agents
    :param truth: [nx3 pandas DF] triples of (x,y,z=f(x,y)) where z=f is the ground truth function at each point
    :param prior: [mx3 pandas DF] triples of (x,y,z~f(x,y)) where z~f is a low-fidelity prior estimate of the ground
                  truth function at each point, and m << n such that the prior estimate is given only at a few points
    :param hyp: [1x9 pandas DF] of log-scaled hyperparameters to use in MFGP model if multi-fidelity simulation
                log-scaled hyp take the form [mu_lo, s^2_lo, L_lo, mu_hi, s^2_hi, L_hi, rho, noise_lo, noise_hi]
                OR
                [1x4 pandas DF] of log-scaled hyperparameters to use in SFGP model if single-fidelity simulation
                log-scaled hyp take the form [mu, s^2, l, noise]
    :param console: boolean indicating whether to display simulation progress on console
    :param plotter: boolean indicating whether to plot simulation progress using plotter.py
    :param log: boolean indicating whether to log simulation progress into CSV for performance analysis
    :return: [3 value tuple] of
        [list of dictionaries] log of loss by iteration
        [list of dictionaries] log of agent positions and actions by iteration
        [list of dictionaries] log of samples taken by agent over the course of simulation
    """
    # 0) initialize logging lists, plotter, and determine fidelity
    loss_log, agent_log, sample_log = [], [], [] if log else None
    plotter.reset() if plotter else None
    if len(hyp.columns) == 4:
        fidelity = "S"  # use singlefidelity model
    elif len(hyp.columns) == 9:
        fidelity = "M"  # use multifidelity model
    else:
        raise TypeError("Hyperparameters must be of length 4 (single-fidelity) or 9 (multi-fidelity)")

    print(line_break + title + line_break) if console else None

    # 1) initialize model with hyperparameters and empty prior
    if fidelity == "S":
        model = init_SFGP(hyp, prior=None)
    else:
        model = init_MFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points, y truth points, loss and bounding box of domain
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [2]]  # all rows, third column is ground truth y points
    bounding_box = np.array([np.amin(x_star[:, 0]), np.amax(x_star[:, 0]),
                             np.amin(x_star[:, 1]), np.amax(x_star[:, 1])])  # [x_min, x_max, y_min, y_max]
    loss = []

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0)) if console else None

    # 4) initialize model with prior and force-update model
    if fidelity == "S":
        model = init_SFGP(hyp, prior=prior)
        model.updt_info(model.X, model.y)
    else:
        model = init_MFGP(hyp, prior=prior)
        model.updt_info(model.X_L, model.y_L, model.X_H, model.y_H)

    # 5) initialize vars and begin iterative portion of algorithm
    iteration = 0
    period = 0
    centroids_t = positions
    prob_explore_t = np.zeros((agents, 1))      # unnecessary for this algorithm, but keep for logging consistency
    explore_t = np.zeros((agents, 1))

    while iteration < iterations:

        # 6) compute prediction and determine threshold below which to reduce uncertainty on this period
        mu_star, var_star = model.predict(x_star)
        max_var_t = np.amax(np.diag(var_star))
        threshold = choi_threshold(var_star)

        # 7) determine partition for each robot to explore based on current Lloyd iteration
        sample_vor = voronoi_bounded(centroids_t, bounding_box)

        # 8) determine points to sample for explore portion of this period
        sample_points = compute_sample_points(model, x_star, threshold, console)

        # 9) k-means cluster points to sample for explore portion of this period
        sample_clusters = compute_sample_clusters(sample_vor, sample_points)

        # 9) determine TSP tours through each cluster
        print("\nBegin TSP Computation") if console else None
        tsp_tours_t = compute_sample_tsp(sample_clusters)   # dynamic list of tours which updates as agents take samples
        print("\nEnd TSP Computation") if console else None
        tsp_tours_0 = copy.deepcopy(tsp_tours_t)            # static list of complete tours for this period

        # 9) determine length of this explore-then-exploit period and execute period
        period_length = choi_double(period)
        for step in range(period_length):

            print(f"\nBegin Iteration {iteration} of Simulation {sim_num} of {title}") if console else None

            # 7) record samples from each agent on an explore step (i.e., on a TSP tour)
            x_new = np.empty([0, 2])    # store new sample points
            y_new = np.empty([0, 1])    # store new samples
            id_new = np.empty([0, 1])   # store agent ids that sampled
            for i in range(agents):
                if explore_t[i] == 1:   # this robot is on an explore step/TSP tour: take sample
                    x_sample = positions[i, :]
                    sample_idx = np.logical_and(truth_arr[:, 0] == x_sample[0], truth_arr[:, 1] == x_sample[1])
                    y_sample = truth_arr[sample_idx, 2]  # retrieve f_val at matching point
                    print(f"Robot {i} explored {x_sample} and sampled {y_sample}") if console else None
                    x_new = np.vstack((x_new, x_sample))
                    y_new = np.vstack((y_new, y_sample))
                    id_new = np.vstack((id_new, i))
                elif iteration > 0:     # 0th iteration is for initialization purposes only
                    print(f"Robot {i} exploited to {centroids_t[i, :]}") if console else None

            # 8) update GP model and estimates (Todescato "Estimate update")
            if fidelity == "S":
                model.updt(x_new, y_new)
            else:
                model.updt_hifi(x_new, y_new)
            mu_star, var_star = model.predict(x_star)

            # 9) compute loss given current positions
            loss_vor = voronoi_bounded(positions, bounding_box)
            loss_t = compute_loss(loss_vor, truth_arr)
            loss.append(loss_t)

            # 10) update partitions and centroids using Lloyd iteration given current estimate
            lloyd_vor = voronoi_bounded(centroids_t, bounding_box)
            centroids_t = compute_centroids(lloyd_vor, x_star, mu_star)

            # 11) update status of learning progress (not necessary for decision, but useful for logging)
            argmax_var_t, max_var_t = compute_max_var(lloyd_vor, truth_arr, var_star)

            # 12) print to console, update log, and plot for this iteration
            # (note: period is logged in all simulations for consistency, and DOES apply here)
            if console:
                print(f"Period {period}")
                print(f"Fidelity {fidelity}")
                print(f"Current loss: {loss_t}")
                print(f"Max var by cell: {max_var_t.flatten()}")
                print(f"Normalizing max var: {max_var_0}")
                print(f"Probability of exploration: {prob_explore_t.flatten()}")
                print(f"Decision of exploration: {explore_t.flatten()}")
                print(f"End Iteration {iteration}")
            if log:
                loss_log.append({"SimNum": sim_num, "Iteration": iteration, "Period": period,
                                 "Fidelity": fidelity, "Loss": loss_t})
                for i in range(agents):
                    agent_log.append({"SimNum": sim_num, "Iteration": iteration, "Period": period,
                                      "Fidelity": fidelity, "Agent": i,
                                      "X": positions[i, 0], "Y": positions[i, 1],
                                      "XMax": argmax_var_t[i, 0], "YMax": positions[i, 1],
                                      "VarMax": max_var_t[i, 0], "Var0": max_var_0,
                                      "XCentroid": centroids_t[i, 0], "YCentroid": centroids_t[i, 1],
                                      "ProbExplore": prob_explore_t[i, 0], "Explore": explore_t[i, 0]})
                for i in range(id_new.size):
                    sample_log.append(
                        {"SimNum": sim_num, "Iteration": iteration, "Period": period, "Fidelity": fidelity,
                         "Agent": id_new[i, 0], "X": x_new[i, 0], "Y": x_new[i, 1], "Sample": y_new[i, 0]})
            if plotter:
                plotter.plot_explore(prob_explore_t, explore_t)
                plotter.plot_mean(x_star, mu_star)
                plotter.plot_var(x_star, var_star)
                plotter.plot_tsp(sample_vor, tsp_tours_0, tsp_tours_t)
                plotter.plot_loss_vor(loss_vor, truth_arr, explore_t)
                plotter.plot_loss(loss)
                plotter.plot_lloyd_vor(lloyd_vor, centroids_t, truth_arr)
                plotter.show()
                # plotter.save(f"Animations/null_two_corners_mf_choi/null_two_corners_mf_choi_{iteration}.png")

            # 13) make explore/exploit decision depending on remaining points in TSP tour for each agent
            for i in range(agents):
                if tsp_tours_t[i].shape[0] > 0:  # this agent still has points to sample from in TSP tour
                    prob_explore_t[i] = 1
                    explore_t[i] = True
                else:  # this agent is done sampling all points in this period's TSP tour
                    prob_explore_t[i] = 0
                    explore_t[i] = False

            # 14) update agent positions and delete points from TSP tour as we go (all points remain in tsp_tour_0)
            for i in range(agents):
                if explore_t[i, 0]:     # take next point in TSP tour, then remove it from the tour
                    positions[i, :] = tsp_tours_t[i][0, :]    # next destination is first point in i-th TSP tour
                    tsp_tours_t[i] = np.delete(tsp_tours_t[i], 0, axis=0)   # delete first row of i-th TSP tour
                else:                   # go to centroid of Lloyd cell
                    positions[i, :] = centroids_t[i, :]

            # 15) increment outer iteration count as we execute each period step
            iteration += 1

        # 16) increment period counter to continue doubling trick
        period += 1

    # 17) return log dictionary lists to driver function, which will save them into a dataframe
    return loss_log, agent_log, sample_log


if __name__ == "__main__":
    """
    Run a series of multiagent learning-coverage algorithm simulations.
    """

    name = "Data/two_corners"           # name of simulation, used as prefix of all associated input filenames
    prefix = "Data/tc256"               # name of simulation, used as prefix of all associated output filenames

    agents = 4              # number of agents to use in simulation
    iterations = 248        # number of iterations to run each simulation
    simulations = 10        # number of simulations to run
    console = True          # boolean indicating if intermediate output should print to console
    log = True              # boolean indicating if output should be logged to CSV for performance analysis
    # plotter = Plotter([-eps, 1 + eps, -eps, 1 + eps])   # x_min, x_max, y_min, y_max
    plotter = None          # do not plot
    np.random.seed(1234)    # seed random generator for reproducibility

    truth = pd.read_csv(f"{name}_hifi.csv")         # CSV specifying ground truth (x,y,z=f(x,y)) triples
    mf_hyp = pd.read_csv(f"{name}_mf_hyp.csv")      # CSV specifying multi-fidelity GP hyperparameters
    sf_hyp = pd.read_csv(f"{name}_sf_hyp.csv")      # CSV specifying single-fidelity GP hyperparameters
    null_prior = pd.read_csv("Data/null_prior.csv")      # Use a null prior
    human_prior = pd.read_csv(f"{name}_prior.csv")        # CSV specifying prior to condition GP upon before simulation

    loss_log, agent_log, sample_log = [], [], []    # Initialize logging lists
    algorithms = ["todescato_nsf", "todescato_hsf", "todescato_hmf",
                  "choi_nsf", "choi_hsf", "choi_hmf"]

    for algo in algorithms:

        out_name = f"{prefix}_{algo}"

        for sim_num in range(simulations):
            print(line_break + f"Simulation {sim_num} : {algo}" + line_break)

            # 1) initialize agent positions
            x_positions = [random.random() for i in range(agents)]
            y_positions = [random.random() for i in range(agents)]
            positions = np.column_stack((x_positions, y_positions))

            # 2) select hyperparameters
            if "mf" in algo:
                hyp = mf_hyp
            else:
                hyp = sf_hyp

            # 3) select prior
            if "_n" in algo:
                prior = null_prior
            else:
                prior = human_prior

            # 4) run simulation
            if "choi" in algo:
                loss_log_t, agent_log_t, sample_log_t = \
                    choi(algo, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)
            else:
                loss_log_t, agent_log_t, sample_log_t = \
                    todescato(algo, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)

            # 3) extend logging lists to include current simulation's results
            loss_log.extend(loss_log_t)
            agent_log.extend(agent_log_t)
            sample_log.extend(sample_log_t)

        # save dataframes from simulation results for post-analysis
        if log:
            loss_df = pd.DataFrame(loss_log)
            loss_df.to_csv(out_name + "_loss.csv")
            agent_df = pd.DataFrame(agent_log)
            agent_df.to_csv(out_name + "_agent.csv")
            sample_df = pd.DataFrame(sample_log)
            sample_df.to_csv(out_name + "_sample.csv")

