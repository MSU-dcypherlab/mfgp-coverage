import sys
import random
import cProfile
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib import path
from gaussian_process import MFGP, SFGP
from plotter import Plotter

eps = 0.1
line_break = "\n" + "".join(["*" for i in range(100)]) + "\n"


def init_MFGP(hyp, prior):
    if prior is not None:
        p = np.vstack(prior.values.tolist())
        X_L = np.reshape(p[:, [0, 1]], (-1,2))  # all rows, first two columns are X,Y of lofi prior
        y_L = np.reshape(p[:, 2], (-1,1))  # all rows, third column is F of lofi prior
    else:
        X_L = np.empty([0, 2])
        y_L = np.empty([0, 1])
    X_H = np.empty([0, 2])
    y_H = np.empty([0, 1])
    len_L = 1
    len_H = 1
    model = MFGP(X_L, y_L, X_H, y_H, len_L, len_H)

    h_arr = np.array(hyp.values.tolist()[0])  # convert hyperparameters from dataframe to list
    model.hyp = h_arr   # convert hyperparameters from list to ndarray

    return model


def init_SFGP(hyp, prior):
    if prior is not None:
        p = np.vstack(prior.values.tolist())
        X = np.reshape(p[:, [0, 1]], (-1, 2))  # all rows, first two columns are X,Y of lofi prior
        y = np.reshape(p[:, 2], (-1, 1))  # all rows, third column is F of lofi prior
    else:
        X = np.empty([0, 2])
        y = np.empty([0, 1])
    len = 1
    model = SFGP(X, y, len)

    h_arr = np.array(hyp.values.tolist()[0])  # convert hyperparameters from dataframe to list
    model.hyp = h_arr  # convert hyperparameters from list to ndarray

    return model


def plot_voronoi(vor, bounding_box):
    # Plot initial points
    plt.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')
    # Plot ridges points
    for region in vor.filtered_regions:
        vertices = vor.vertices[region, :]
        plt.plot(vertices[:, 0], vertices[:, 1], 'go')
    # Plot ridges
    for region in vor.filtered_regions:
        vertices = vor.vertices[region + [region[0]], :]
        plt.plot(vertices[:, 0], vertices[:, 1], 'k-')
    plt.xlim(bounding_box[0] - 0.1, bounding_box[1] + 0.1)
    plt.ylim(bounding_box[2] - 0.1, bounding_box[3] + 0.1)
    plt.show()


def in_polygon(xq, yq, xv, yv):
    """
    Retrived from https://stackoverflow.com/a/49733403 May 23, 2020
    Translates Matlab inpolygon implementation described here: https://www.mathworks.com/help/matlab/ref/inpolygon.html
    Determines if query points (xq, yq) are contained in the polygon specified by (xv, yv)
    :param xq:
    :param yq:
    :param xv:
    :param yv:
    :return:
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
    Retrieved from https://stackoverflow.com/a/30408825 May 24, 2020
    Implementation of the Shoelace formula for computing polygonal area (https://en.wikipedia.org/wiki/Shoelace_formula)
    :param x:
    :param y:
    :return:
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def in_box(points, bounding_box):
    """
    Retrieved from https://stackoverflow.com/a/33602171 May 23, 2020
    :param points: 
    :param bounding_box: 
    :return: 
    """
    return np.logical_and(np.logical_and(bounding_box[0] - eps <= points[:, 0],
                                         points[:, 0] <= bounding_box[1] + eps),
                          np.logical_and(bounding_box[2] - eps <= points[:, 1],
                                         points[:, 1] <= bounding_box[3] + eps))


def voronoi_bounded(points, bounding_box):
    """
    Retrieved from https://stackoverflow.com/a/33602171 May 23, 2020
    :param points:
    :param bounding_box:
    :return:
    """
    # Select points inside the bounding box
    i = in_box(points, bounding_box)
    # Mirror points
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
    # Compute Voronoi
    vor = Voronoi(points)
    # Filter regions
    regions = []
    for region in vor.regions:
        flag = True
        for index in region:
            if index == -1:
                flag = False
                break
            else:
                x = vor.vertices[index, 0]
                y = vor.vertices[index, 1]
                if not(bounding_box[0] - eps <= x <= bounding_box[1] + eps and
                       bounding_box[2] - eps <= y <= bounding_box[3] + eps):
                    flag = False
                    break
        if region != [] and flag:
            regions.append(region)
    vor.filtered_points = points_center
    vor.filtered_regions = np.array(vor.regions)[vor.point_region[:vor.npoints//5]]
    return vor


def compute_loss(vor, truth_arr):
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

        # plt.plot(in_points[:,0], in_points[:,1], 'ro')
        # plt.plot(center[0], center[1], 'go')
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    # 4) return loss summed over all cells
    return loss


def compute_centroids(vor, x_star, mu_star):
    centroids = np.empty([0, 2])

    # 1) iterate over each cell in voronoi partition and compute centroid
    for i, cell in enumerate(vor.filtered_regions):
        # 2) select only the truth points in this cell from truth_arr
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
        centroid = weighted_integral / f_integral   # mean 1x2 weighted location of cell
        centroids = np.vstack((centroids, centroid))

        plt.figure()
        plt.scatter(in_points[:, 0], in_points[:, 1], c=in_means[:, 0])
        plt.plot(centroid[0], centroid[1], 'k+')
        plt.xlim((-0.1, 1.1))
        plt.ylim((-0.1, 1.1))
        plt.show()

    return centroids


def compute_max_var(vor, truth_arr, var_star):
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
        argmax_var = in_points[in_var == max_var, :][0, [0, 1]]    # pick first of multiple argmax's and take (x,y) only
        argmax_var_t = np.vstack((argmax_var_t, argmax_var))
        max_var_t = np.vstack((max_var_t, max_var))

        # plt.scatter(in_points[:, 0], in_points[:, 1], c=in_var)
        # plt.plot(argmax_var[0], argmax_var[1], 'k+')
        # plt.xlim((-0.1, 1.1))
        # plt.ylim((-0.1, 1.1))
        # plt.show()

    return argmax_var_t, max_var_t

def sfgp_todescato(sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log):

    print(line_break + "SFGP Todescato" + line_break) if console else None

    # 0) Initialize logging dict-lists
    if log:
        loss_log, agent_log, sample_log, gp_log = [], [], [], []

    # 1) initialize SFGP model with hyperparameters and empty prior
    model = init_SFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points, y truth points, loss and bounding box of domain
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [0, 1]]  # all rows, third column is ground truth y points
    bounding_box = np.array([np.amin(x_star[:, 0]), np.amax(x_star[:, 0]),
                             np.amin(x_star[:, 1]), np.amax(x_star[:, 1])])  # [x_min, x_max, y_min, y_max]
    loss = []

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0)) if console else None

    # 4) initialize SFGP model with prior to force-update model
    model = init_SFGP(hyp, prior=prior)

    # 5) compute prediction given prior and initialize relevant explore/exploit decision variables
    mu_star, var_star = model.predict(x_star)
    var = np.diag(var_star)
    max_var_t = np.amax(var) * np.ones((agents, 1))
    prob_explore_t = max_var_t / max_var_0 * np.ones((agents, 1))
    explore_t = np.zeros((agents, 1))   # initialize to zero so agents do not sample on first iteration
    centroids_t = positions     # initialize centroids governing Lloyd iterations to current positions

    # 6) begin iterative portion of algorithm
    for iteration in range(iterations):

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
        model.updt(x_new, y_new)
        mu_star, var_star = model.predict(x_star)

        # 9) compute loss given current positions
        loss_vor = voronoi_bounded(positions, bounding_box)
        loss_t = compute_loss(loss_vor, truth_arr)
        loss.append(loss_t)

        # 10) update partitions and centroids given current estimate (Todescato "Partition and centroids update")
        lloyd_vor = voronoi_bounded(centroids_t, bounding_box)
        centroids_t = compute_centroids(lloyd_vor, x_star, mu_star)

        # 11) compute points of max variance and make explore/exploit decision (Todescato "Target-Points computation")
        argmax_var_t, max_var_t = compute_max_var(lloyd_vor, truth_arr, var_star)
        prob_explore_t = max_var_t / max_var_0
        explore_t = np.array([random.random() < cutoff for cutoff in prob_explore_t])    # Bernoulli wrt prob_explore_t

        # 12) print to console, update log, and plot for this iteration
        if console:
            print(f"\nIteration {iteration}")
            print(f"Current loss: {loss_t}")
            print(f"Max var by cell: {max_var_t.flatten()}")
            print(f"Normalizing max var: {max_var_0}")
            print(f"Probability of exploration: {prob_explore_t.flatten()}")
            print(f"Decision of exploration: {explore_t.flatten()}")
        if log:
            loss_log.append({"SimNum": sim_num, "Iteration": iteration, "Loss": loss_t})
            for i in range(agents):
                agent_log.append({"SimNum": sim_num, "Iteration": iteration, "Agent": i,
                                  "X": positions[i, 0], "Y": positions[i, 1],
                                  "XMax": argmax_var_t[i, 0], "YMax": positions[i, 1],
                                  "VarMax": max_var_t[i, 0], "Var0": max_var_0,
                                  "XCentroid": centroids_t[i, 0], "YCentroid": centroids_t[i, 1],
                                  "ProbExplore": prob_explore_t[i, 0], "Explore": explore_t[i, 0]})
            for i in range(id_new.size):
                sample_log.append({"SimNum": sim_num, "Iteration": iteration, "Agent": id_new[i, 0],
                                   "X": x_new[i, 0], "Y": x_new[i, 1], "Sample": y_new[i, 0]})
            # var = np.diag(var_star)
            # for i in range(x_star.shape[0]):
            #     gp_log.append({"SimNum": sim_num, "Iteration": iteration,
            #                    "X": x_star[i, 0], "Y": x_star[i, 1],
            #                    "Mu": mu_star[i, 0], "Var": var[i]})
        if plotter:
            plotter.plot_explore(prob_explore_t, explore_t)
            plotter.plot_mean(x_star, mu_star)
            plotter.plot_var(x_star, var_star)
            plotter.plot_loss_vor(loss_vor, truth_arr, explore_t)
            plotter.plot_loss(loss)
            plotter.plot_lloyd_vor(lloyd_vor, centroids_t, truth_arr)
            plotter.show()

        # 13) update agent positions (Todescato "Target-Points transmission")
        for i in range(agents):
            if explore_t[i, 0]:
                positions[i, :] = argmax_var_t[i, :]
            else:
                positions[i, :] = centroids_t[i, :]

    # 14) return log dictionary lists to driver function, which will save them into a dataframe
    return loss_log, agent_log, sample_log, gp_log

def mfgp_todescato(sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log):

    print(line_break + "MFGP Todescato" + line_break) if console else None

    # 0) Initialize logging dict-lists
    if log:
        loss_log, agent_log, sample_log, gp_log = [], [], [], []

    # 1) initialize MFGP model with hyperparameters and empty prior
    model = init_MFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points, y truth points, loss and bounding box of domain
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [0, 1]]  # all rows, third column is ground truth y points
    bounding_box = np.array([np.amin(x_star[:, 0]), np.amax(x_star[:, 0]),
                             np.amin(x_star[:, 1]), np.amax(x_star[:, 1])])  # [x_min, x_max, y_min, y_max]
    loss = []

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0)) if console else None

    # 4) initialize MFGP model with prior and force-update model
    model = init_MFGP(hyp, prior=prior)
    model.updt_info(model.X_L, model.y_L, model.X_H, model.y_H)

    # 5) compute prediction given prior and initialize relevant explore/exploit decision variables
    mu_star, var_star = model.predict(x_star)
    var = np.diag(var_star)
    max_var_t = np.amax(var) * np.ones((agents, 1))
    prob_explore_t = max_var_t / max_var_0 * np.ones((agents, 1))
    explore_t = np.zeros((agents, 1))   # initialize to zero so agents do not sample on first iteration
    centroids_t = positions     # initialize centroids governing Lloyd iterations to current positions

    # 6) begin iterative portion of algorithm
    for iteration in range(iterations):

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
        model.updt_hifi(x_new, y_new)
        mu_star, var_star = model.predict(x_star)

        # 9) compute loss given current positions
        loss_vor = voronoi_bounded(positions, bounding_box)
        loss_t = compute_loss(loss_vor, truth_arr)
        loss.append(loss_t)

        # 10) update partitions and centroids given current estimate (Todescato "Partition and centroids update")
        lloyd_vor = voronoi_bounded(centroids_t, bounding_box)
        centroids_t = compute_centroids(lloyd_vor, x_star, mu_star)

        # 11) compute points of max variance and make explore/exploit decision (Todescato "Target-Points computation")
        argmax_var_t, max_var_t = compute_max_var(lloyd_vor, truth_arr, var_star)
        prob_explore_t = max_var_t / max_var_0
        explore_t = np.array([random.random() < cutoff for cutoff in prob_explore_t])    # Bernoulli wrt prob_explore_t

        # 12) print to console, update log, and plot for this iteration
        if console:
            print(f"\nIteration {iteration}")
            print(f"Current loss: {loss_t}")
            print(f"Max var by cell: {max_var_t.flatten()}")
            print(f"Normalizing max var: {max_var_0}")
            print(f"Probability of exploration: {prob_explore_t.flatten()}")
            print(f"Decision of exploration: {explore_t.flatten()}")
        if log:
            loss_log.append({"SimNum": sim_num, "Iteration": iteration, "Loss": loss_t})
            for i in range(agents):
                agent_log.append({"SimNum": sim_num, "Iteration": iteration, "Agent": i,
                                  "X": positions[i, 0], "Y": positions[i, 1],
                                  "XMax": argmax_var_t[i, 0], "YMax": positions[i, 1],
                                  "VarMax": max_var_t[i, 0], "Var0": max_var_0,
                                  "XCentroid": centroids_t[i, 0], "YCentroid": centroids_t[i, 1],
                                  "ProbExplore": prob_explore_t[i, 0], "Explore": explore_t[i, 0]})
            for i in range(id_new.size):
                sample_log.append({"SimNum": sim_num, "Iteration": iteration, "Agent": id_new[i, 0],
                                   "X": x_new[i, 0], "Y": x_new[i, 1], "Sample": y_new[i, 0]})
            # var = np.diag(var_star)
            # for i in range(x_star.shape[0]):
            #     gp_log.append({"SimNum": sim_num, "Iteration": iteration,
            #                    "X": x_star[i, 0], "Y": x_star[i, 1],
            #                    "Mu": mu_star[i, 0], "Var": var[i]})
        if plotter:
            plotter.plot_explore(prob_explore_t, explore_t)
            plotter.plot_mean(x_star, mu_star)
            plotter.plot_var(x_star, var_star)
            plotter.plot_loss_vor(loss_vor, truth_arr, explore_t)
            plotter.plot_loss(loss)
            plotter.plot_lloyd_vor(lloyd_vor, centroids_t, truth_arr)
            plotter.show()

        # 13) update agent positions (Todescato "Target-Points transmission")
        for i in range(agents):
            if explore_t[i, 0]:
                positions[i, :] = argmax_var_t[i, :]
            else:
                positions[i, :] = centroids_t[i, :]

    # 14) return log dictionary lists to driver function, which will save them into a dataframe
    return loss_log, agent_log, sample_log, gp_log


if __name__ == "__main__":

    name = "Data/sf_fc_false"
    agents = 4
    iterations = 100
    simulations = 10
    console = True
    plotter = None #Plotter([-0.1, 1.1, -0.1, 1.1])   # x_min, x_max, y_min, y_max
    log = True

    truth = pd.read_csv(name + "_truth.csv")
    hyp = pd.read_csv(name + "_hyp.csv")
    prior = pd.read_csv(name + "_prior.csv")

    loss_log, agent_log, sample_log, gp_log = [], [], [], []

    for sim_num in range(simulations):
        print(line_break + f"Simulation {sim_num}" + line_break)

        x_positions = [.83, .80, .65, .06]#[random.random() for i in range(agents)]
        y_positions = [.93, .81, .81, .37]#[random.random() for i in range(agents)]
        positions = np.column_stack((x_positions, y_positions))

        loss_log_t, agent_log_t, sample_log_t, gp_log_t = \
            sfgp_todescato(sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)
            # mfgp_todescato(sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)
        loss_log.extend(loss_log_t)
        agent_log.extend(agent_log_t)
        sample_log.extend(sample_log_t)
        # gp_log.extend(gp_log_t)

    # save dataframes from simulation results
    loss_df = pd.DataFrame(loss_log)
    loss_df.to_csv(name + "_loss.csv")
    agent_df = pd.DataFrame(agent_log)
    agent_df.to_csv(name + "_agent.csv")
    sample_df = pd.DataFrame(sample_log)
    sample_df.to_csv(name + "_sample.csv")
    # gp_df = pd.DataFrame(gp_log)
    # gp_df.to_csv(name + "_gp.csv")
