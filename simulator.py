import sys
import random
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib import path
from gaussian_process import MFGP, SFGP

eps = sys.float_info.epsilon
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

def init_SFGP(hyp):
    # TODO: Implement SFGP
    pass

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


def in_box(points, bounding_box):
    """
    Retrieved from https://stackoverflow.com/a/33602171 May 23, 2020
    :param points: 
    :param bounding_box: 
    :return: 
    """
    return np.logical_and(np.logical_and(bounding_box[0] <= points[:, 0],
                                         points[:, 0] <= bounding_box[1]),
                          np.logical_and(bounding_box[2] <= points[:, 1],
                                         points[:, 1] <= bounding_box[3]))


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
    points_left[:, 0] = bounding_box[0] - (points_left[:, 0] - bounding_box[0])
    points_right = np.copy(points_center)
    points_right[:, 0] = bounding_box[1] + (bounding_box[1] - points_right[:, 0])
    points_down = np.copy(points_center)
    points_down[:, 1] = bounding_box[2] - (points_down[:, 1] - bounding_box[2])
    points_up = np.copy(points_center)
    points_up[:, 1] = bounding_box[3] + (bounding_box[3] - points_up[:, 1])
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
        point_loss = distances * in_points[:, 2]    # loss_of_each_point = dist * f_val
        cell_loss = np.mean(point_loss)  # loss_of_each_cell = integral(loss_of_each_point), discretized as average
        loss += cell_loss

        # plt.plot(in_points[:,0], in_points[:,1], 'ro')
        # plt.plot(center[0], center[1], 'go')
        # plt.xlim((0,1))
        # plt.ylim((0,1))
        # plt.show()

    # 4) return loss summed over all cells
    return loss


def mfgp_todescato(name, iterations, agents, positions, truth, prior, hyp, plot=False):
    print(line_break + "MFGP Todescato" + line_break)

    # 1) initialize MFGP model with hyperparameters and empty prior
    model = init_MFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points, y truth points and bounding box of domain
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [0, 1]]  # all rows, third column is ground truth y points
    bounding_box = np.array([np.amin(x_star[:, 0]), np.amax(x_star[:, 0]),
                             np.amin(x_star[:, 1]), np.amax(x_star[:, 1])])  # [x_min, x_max, y_min, y_max]

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0))

    # 4) initialize MFGP model with prior and force-update model
    model = init_MFGP(hyp, prior=prior)
    model.updt_info(model.X_L, model.y_L, model.X_H, model.y_H)

    # 5) compute prediction given prior initialize relevant explore/exploit decision variables
    mu_star, var_star = model.predict(x_star)
    max_var_t = np.amax(var_star) * np.ones((agents, 1))
    prob_explore_t = max_var_t / max_var_0 * np.ones((agents, 1))
    explore_t = np.zeros((agents, 1))   # initialize to zero so agents do not sample on first iteration

    # 6) begin iterative portion of algorithm
    for iteration in range(iterations):
        print(f"Iteration {iteration}")

        # 7) record samples from each agent on explore step (Todescato "Listen")
        for i in range(len(agents)):
            if prob_explore_t[i] == 1:  # this robot is on an explore step: take sample
                # take sample
                print('sample')

        # 8) update GP estimates (Todescato "Estimate update")

        # 9) compute loss given current positions
        loss_vor = voronoi_bounded(positions, bounding_box)
        plot_voronoi(loss_vor, bounding_box)
        loss_t = compute_loss(positions, loss_vor, truth_arr)

        # 10) update partitions and centroids (Todescato "Partition and centroids update")

        # 11) compute points of max variance and make explore/exploit decision (Todescato "Target-Points computation")

        # 12) update agent positions (Todescato "Target-Points transmission"


        # TODO: finish Todescato MFGP


if __name__ == "__main__":

    name = "ex"
    agents = 4
    iterations = 1
    simulations = 1
    plot = False

    truth = pd.read_csv(name + "_truth.csv")
    hyp = pd.read_csv(name + "_hyp.csv")
    prior = pd.read_csv(name + "_prior.csv")

    for sim_num in range(simulations):
        print(line_break + f"Simulation {sim_num}" + line_break)

        x_positions = [random.random() for i in range(agents)]
        y_positions = [random.random() for i in range(agents)]
        positions = np.column_stack((x_positions, y_positions))

        mfgp_todescato(name, iterations, agents, positions, truth, prior, hyp, plot)

