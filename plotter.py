import matplotlib.pyplot as plt
import numpy as np


class Plotter:
    #__slots__ = ["Fig, Bounds, LloydAx, LossVorAx, MeanAx, VarAx, LossAx, ProbAx"]

    def __init__(self, bounding_box):
        """
        Referenced https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html May 24, 2020
        :param bounding_box:
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
        self.LossAx.cla()
        self.LossAx.plot(self.Iterations, loss, 'ko-')
        self.LossAx.set_title("Loss")
        self.Iteration += 1
        self.Iterations.append(self.Iteration)

    def plot_mean(self, x_star, mean):
        self.MeanAx.cla()
        self.MeanAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.MeanAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.MeanAx.set_aspect("equal")
        self.MeanAx.set_title("Posterior Mean")

        self.MeanAx.scatter(x_star[:, 0], x_star[:, 1], c=mean[:, 0])

    def plot_var(self, x_star, var_star):
        var = np.diag(var_star)  # point variances are diagonal of cov matrix

        self.VarAx.cla()
        self.VarAx.set_xlim(self.Bounds[0], self.Bounds[1])
        self.VarAx.set_ylim(self.Bounds[2], self.Bounds[3])
        self.VarAx.set_aspect("equal")
        self.VarAx.set_title("Posterior Var")

        self.VarAx.scatter(x_star[:, 0], x_star[:, 1], c=var)

    def plot_explore(self, prob_explore, explore):
        self.ExpAx.cla()
        self.ExpAx.set_title("Probability of Exploration")
        self.ExpAx.set_ylim(self.Bounds[2], self.Bounds[3])

        color = ['r' if e else 'k' for e in explore.flatten()]
        self.ExpAx.bar(x=range(prob_explore.size), height=prob_explore[:, 0])
        self.ExpAx.scatter(x=range(explore.size), y=explore[:, 0], c=color)

    def show(self):
        self.Fig.show()

    def reset(self):
        self.Iteration = 0
        self.Iterations = [0]
