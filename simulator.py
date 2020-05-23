import random
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from gaussian_process import MFGP, SFGP


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

def mfgp_todescato(name, iterations, agents, positions, truth, prior, hyp):
    print(line_break + "MFGP Todescato" + line_break)

    # 1) initialize MFGP model with hyperparameters and empty prior
    model = init_MFGP(hyp, prior=None)

    # 2) initialize arrays of x_star test points and y truth points
    truth_arr = np.vstack(truth.values.tolist())
    x_star = truth_arr[:, [0, 1]]  # all rows, first two columns are X* gridded test points
    y = truth_arr[:, [0, 1]]  # all rows, third column is ground truth y points

    # 3) compute max predictive variance and keep as normalizing constant
    mu_star, var_star = model.predict(x_star)
    max_var_0 = np.amax(var_star)
    print("Max Initial Predictive Variance: " + str(max_var_0))

    # 4) initialize MFGP model with prior and force-update model
    model = init_MFGP(hyp, prior=prior)
    model.updt_info(model.X_L, model.y_L, model.X_H, model.y_H)

    # 5) compute prediction given prior and store updated max predictive variance
    mu_star, var_star = model.predict(x_star)
    max_var_t = np.amax(var_star)

    # 6) begin iterative portion of algorithm
    for iteration in range(iterations):
        print(f"Iteration {iteration}")

        # 7) compute voronoi cells of agents given current positions
        vor = Voronoi(positions)
        voronoi_plot_2d(vor)
        plt.show()

        # TODO: finish Todescato MFGP


if __name__ == "__main__":

    name = "ex"
    agents = 4
    iterations = 100
    simulations = 1

    truth = pd.read_csv(name + "_truth.csv")
    hyp = pd.read_csv(name + "_hyp.csv")
    prior = pd.read_csv(name + "_prior.csv")

    for sim_num in range(simulations):
        print(line_break + f"Simulation {sim_num}" + line_break)

        x_positions = [random.random() for i in range(agents)]
        y_positions = [random.random() for i in range(agents)]
        positions = np.column_stack((x_positions, y_positions))

        mfgp_todescato(name, iterations, agents, positions, truth, prior, hyp)

