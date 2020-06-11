"""
plotter.py

Plots progression of learning-coverage algorithms and 3-dimensional (x, y, z=f(x,y)) distributions using matplotlib.

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/11/2020
"""

import matplotlib.pyplot as plt
import numpy as np


class Plotter:

    def __init__(self, bounding_box):
        """
        Initialize the plotting canvas with named subplot axes.
        Referenced https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html May 24, 2020

        :param bounding_box: [1x4 numpy array] containing limits [x_min, x_max, y_min, y_max]
        """
        self.Fig = plt.figure(num=0, figsize=(6, 8), dpi=80, facecolor='w')
        self.Fig.subplots_adjust(hspace=0.4, wspace=0.4)
        self.Bounds = bounding_box
        self.Iteration = 0
        self.Iterations = [0]

        self.LloydAx = self.Fig.add_subplot(3, 2, 1)
        self.LloydAx.set_title("Lloyd Iteration")
        self.LloydAx.set_xlim(bounding_box[0], bounding_box[1])
        self.LloydAx.set_ylim(bounding_box[2], bounding_box[3])

        self.LossVorAx = self.Fig.add_subplot(3, 2, 2)
        self.LossVorAx.set_title("Current Loss Cells")
        self.LossVorAx.set_xlim(bounding_box[0], bounding_box[1])
        self.LossVorAx.set_ylim(bounding_box[2], bounding_box[3])

        self.MeanAx = self.Fig.add_subplot(3, 2, 3)
        self.MeanAx.set_title("Posterior Mean")
        self.MeanAx.set_xlim(bounding_box[0], bounding_box[1])
        self.MeanAx.set_ylim(bounding_box[2], bounding_box[3])

        self.VarAx = self.Fig.add_subplot(3, 2, 4)
        self.VarAx.set_title("Posterior Var")
        self.VarAx.set_xlim(bounding_box[0], bounding_box[1])
        self.VarAx.set_ylim(bounding_box[2], bounding_box[3])

        self.LossAx = self.Fig.add_subplot(3, 2, 5)
        self.LossAx.set_title("Loss")

        self.ExpAx = self.Fig.add_subplot(3, 2, 6)
        self.ExpAx.set_title("Probability of Exploration")

    def plot_loss_vor(self, vor, truth, explore):
        """
        Plot bounded Voronoi diagram corresponding to partition of latest loss calculation.
        Loss given by Equation 2 of Todescato et. al. "Multi-robots Gaussian estimation and coverage..."
        Referenced https://stackoverflow.com/a/33602171 May 24, 2020 for plotting bounded Voronoi

        :param vor: [scipy Voronoi object] bounded Voronoi partition corresponding to loss calculation to be plotted,
                    with bounded points and regions specified in filtered_points and filtered_regions fields
        :param truth: [nx3 numpy array] of (x,y,z) triples where z=f(x,y) is the ground truth function at each point
        :param explore: [nAgentsx1 numpy array] of boolean variables indicating if robot i is on explore step
        :return: None
        """
        self.LossVorAx.cla()
        self.LossVorAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.LossVorAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.LossVorAx.set_aspect("equal")
        self.LossVorAx.set_title("Loss Cells")

        # Plot ground truth
        self.LossVorAx.scatter(truth[:, 0], truth[:, 1], c=truth[:, 2], alpha=0.1)

        # Plot center points colored according to explore/exploit
        color = ['r' if e else 'k' for e in explore.flatten()]
        self.LossVorAx.scatter(vor.filtered_points[:, 0], vor.filtered_points[:, 1], c=color)
        # Plot ridge points
        # for region in vor.filtered_regions:
        #     vertices = vor.vertices[region, :]
        #     self.LossVorAx.plot(vertices[:, 0], vertices[:, 1], 'wo')
        # Plot ridges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            self.LossVorAx.plot(vertices[:, 0], vertices[:, 1], 'w-')

    def plot_lloyd_vor(self, vor, centroids, truth):
        """
        Plot bounded Voronoi diagram corresponding to partition of latest Lloyd iteration.
        Lloyd's Algorithm given by Equation 3 of Todescato et. al. "Multi-robots Gaussian estimation and coverage..."
        Referenced https://stackoverflow.com/a/33602171 May 24, 2020 for plotting bounded Voronoi

        :param vor: [scipy Voronoi object] Voronoi partition corresponding to latest Lloyd iteration to be plotted
        :param centroids: [nAgentsx2 numpy array] of (x,y) pairs indicating each robot's cell centroid
        :param truth: [nx3 numpy array] of (x,y,z) triples where z=f(x,y) is the ground truth function at each point
        :return: None
        """
        self.LloydAx.cla()
        self.LloydAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.LloydAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.LloydAx.set_aspect("equal")
        self.LloydAx.set_title("Lloyd Cells")

        # Plot ground truth
        self.LloydAx.scatter(truth[:, 0], truth[:, 1], c=truth[:, 2], alpha=0.1)

        # Plot center points
        self.LloydAx.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'ko')
        # Plot weighted centroids
        self.LloydAx.plot(centroids[:, 0], centroids[:, 1], 'r+')

        # Plot ridge points
        # for region in vor.filtered_regions:
        #     vertices = vor.vertices[region, :]
        #     self.LloydAx.plot(vertices[:, 0], vertices[:, 1], 'wo')
        # Plot ridges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            self.LloydAx.plot(vertices[:, 0], vertices[:, 1], 'w-')

    def plot_loss(self, loss):
        """
        Plot loss of current iteration along with loss of all previous iterations. Increment iteration of plotter.

        :param loss: [scalar] current loss to be added to loss-vs-iteration plot
        :return: None
        """
        self.LossAx.cla()
        self.LossAx.plot(self.Iterations, loss, 'ko-')
        self.LossAx.set_title("Loss")
        self.Iteration += 1
        self.Iterations.append(self.Iteration)

    def plot_mean(self, x_star, mean):
        """
        Plot posterior mean estimate over input space defined by x_star.

        :param x_star: [nx2 numpy array] of (x,y) points at which posterior mean was estimated
        :param mean: [nx1 numpy array] of mu(x,y) estimates of posterior mean
        :return: None
        """
        self.MeanAx.cla()
        self.MeanAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.MeanAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.MeanAx.set_aspect("equal")
        self.MeanAx.set_title("Posterior Mean")

        self.MeanAx.scatter(x_star[:, 0], x_star[:, 1], c=mean[:, 0])

    def plot_var(self, x_star, var_star):
        """
        Plot posterior variance estimate over input space defined by x_star.

        :param x_star: [nx2 numpy array] of (x,y) points at which posterior variance was estimated
        :param var_star: [nxn numpy array] of cov(x,x') estimates of posterior variance (diagonal contains variances)
        :return: None
        """
        var = np.diag(var_star)  # point variances are diagonal of cov matrix

        self.VarAx.cla()
        self.VarAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.VarAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.VarAx.set_aspect("equal")
        self.VarAx.set_title("Posterior Var")

        self.VarAx.scatter(x_star[:, 0], x_star[:, 1], c=var)

    def plot_explore(self, prob_explore, explore):
        """
        Plot the current probability of exploration and explore/exploit decision for each agent.

        :param prob_explore: [nRobotx1 numpy array] of the probability of robot i exploring on this step
        :param explore: [nAgentsx1 numpy array] of boolean variables indicating if robot i is on explore step
        :return: None
        """
        self.ExpAx.cla()
        self.ExpAx.set_title("Probability of Exploration")
        self.ExpAx.set_ylim(self.Bounds[2], self.Bounds[3])

        color = ['r' if e else 'k' for e in explore.flatten()]
        self.ExpAx.bar(x=range(prob_explore.size), height=prob_explore[:, 0])
        self.ExpAx.scatter(x=range(explore.size), y=explore[:, 0], c=color)

    def show(self):
        """
        Show canvas of subplots.

        :return: None
        """
        self.Fig.show()

    def reset(self):
        """
        Reset plotter object between simulations. Clear iteration counter and list used in loss-vs-iteration plot.

        :return: None
        """
        self.Iteration = 0
        self.Iterations = [0]
