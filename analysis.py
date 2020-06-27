"""
analysis.py

Scripts to be used in post-processing of log files. Generates visualizations and computes various metrics to
evaluate algorithm performance.

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/17/2020
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

light_rgb = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
dark_rgb = [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
colors = light_rgb + dark_rgb


def compute_dist(agents):
    """

    :param agents:
    :return:
    """
    for title, df in agents.items():

        # compute difference from last iteration, fixing sim and agent
        diffs = df.groupby(by=["SimNum", "Agent"]).diff()

        # compute distance traveled as dx^2 + dy^2
        df["Distance2"] = np.sqrt(diffs["X"] ** 2 + diffs["Y"] ** 2)
        df = df.fillna(0)


def plot_loss(losses, name=None):
    """

    :param losses:
    :param name:
    :return:
    """
    loss_df = None
    for title, df in losses.items():

        # compute average of loss over iterations
        mean_loss = df.groupby(by="Iteration").mean()
        mean_loss = pd.DataFrame(mean_loss["Loss"])
        mean_loss.columns = [title]

        # concatenate onto df of all simulation losses (except first iteration)
        if loss_df is not None:
            loss_df = pd.concat([loss_df, mean_loss], axis=1)
        else:
            loss_df = mean_loss

    # plot df of all simulation losses
    plt.figure()
    loss_df.plot(color=colors)
    plt.title("Loss by Iteration")
    plt.savefig(f"Images/{name}_loss.png") if name is not None else None
    plt.show()

    # plot closeup of simulation loss at start
    trunc = loss_df[loss_df.index <= 50]
    plt.figure()
    trunc.plot(color=colors)
    plt.title("Loss by Iteration: Zoomed")
    plt.savefig(f"Images/{name}_loss_zoomed.png") if name is not None else None
    plt.show()

    # compute rolling loss and plot
    window_size = 10
    rolling_loss_df = loss_df.rolling(window_size, min_periods=1, center=True).mean()
    plt.figure()
    rolling_loss_df.plot(color=colors)
    plt.title("Moving Average Loss by Iteration (n=10)")
    plt.savefig(f"Images/{name}_loss_rolling.png") if name is not None else None
    plt.show()


def plot_regret(losses, name=None):
    """

    :param losses:
    :param name:
    :return:
    """
    regret_df = None
    for title, df in losses.items():

        # compute average of loss over iterations
        mean_loss = df.groupby(by="Iteration").mean()
        mean_loss = pd.DataFrame(mean_loss["Loss"])
        mean_loss.columns = [title]

        # subtract min value from all entries
        last_loss = mean_loss.iloc[-1]
        min_loss = mean_loss.min()
        norm_loss = np.maximum(mean_loss - min_loss, 0)

        # cumulative-sum normalized loss to obtain regret
        regret = norm_loss.cumsum()

        # concatenate onto df of all simulation regrets (except first iteration)
        if regret_df is not None:
            regret_df = pd.concat([regret_df, regret], axis=1)
        else:
            regret_df = regret

    # plot df of all simulation regrets
    plt.figure()
    regret_df.plot(color=colors)
    plt.title("Regret by Iteration")
    plt.savefig(f"Images/{name}_regret.png") if name is not None else None
    plt.show()
    #
    # # compute rolling loss and plot
    # window_size = 10
    # rolling_loss_df = loss_df.rolling(window_size, min_periods=1, center=True).mean()
    # plt.figure()
    # rolling_loss_df.plot()
    # plt.title("Moving Average Loss by Iteration (n=10)")


def plot_var(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    var_df = None
    for title, df in agents.items():

        # compute average of loss over iterations
        max_var_by_sim = df.groupby(by=["SimNum", "Iteration"]).max()
        max_var_by_itr = max_var_by_sim.groupby(by="Iteration").mean()
        max_var = pd.DataFrame(max_var_by_itr["VarMax"])
        max_var.columns = [title]

        # concatenate onto df of all simulation vars (except first iteration)
        if var_df is not None:
            var_df = pd.concat([var_df, max_var], axis=1)
        else:
            var_df = max_var

    # plot df of all simulation vars
    plt.figure()
    var_df.plot(color=colors)
    plt.title("Max Variance by Iteration")
    plt.savefig(f"Images/{name}_var.png") if name is not None else None
    plt.show()


def plot_explore(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    explore_df = None
    for title, df in agents.items():

        # compute average of loss over iterations
        prob_explore = df.groupby(by="Iteration").mean()
        prob_explore = pd.DataFrame(prob_explore["ProbExplore"])
        prob_explore.columns = [title]

        # concatenate onto df of all simulation explorations (except first iteration)
        if explore_df is not None:
            explore_df = pd.concat([explore_df, prob_explore], axis=1)
        else:
            explore_df = prob_explore

    # plot df of all simulation explorations
    plt.figure()
    explore_df.plot(color=colors)
    plt.title("Probability of Exploration by Iteration")
    plt.savefig(f"Images/{name}_explore.png") if name is not None else None
    plt.show()



def plot_dist(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    dist_df = None
    for title, df in agents.items():

        # sum distance traveled over all agents for fixed simnum, iteration
        distance_sum = df.groupby(by=["SimNum", "Iteration"]).sum()

        # average distance by all agents per iteration for fixed iteration
        distance = distance_sum.groupby(by="Iteration").mean()

        distance = pd.DataFrame(distance["Distance"])
        distance.columns = [title]

        # concatenate onto df of all distances traveled (except first iteration)
        if dist_df is not None:
            dist_df = pd.concat([dist_df, distance], axis=1)
        else:
            dist_df = distance

    # plot df of all distances traveled
    plt.figure()
    dist_df.plot(color=colors)
    plt.title("Distance Travelled by Iteration")
    plt.savefig(f"Images/{name}_dist.png") if name is not None else None
    plt.show()


def plot_total_dist(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    dist_df = None
    for title, df in agents.items():

        # sum distance traveled over all agents for fixed simnum, iteration
        distance_sum = df.groupby(by=["SimNum", "Iteration"]).sum()

        # average distance by all agents per iteration for fixed iteration
        distance = distance_sum.groupby(by="Iteration").mean()

        # compute total distance as cumulative sum of distance per iteration
        total_distance = pd.DataFrame(distance["Distance"].cumsum())
        total_distance.columns = [title]

        # concatenate onto df of all total distances traveled (except first iteration)
        if dist_df is not None:
            dist_df = pd.concat([dist_df, total_distance], axis=1)
        else:
            dist_df = total_distance

    # plot df of all total distances traveled
    plt.figure()
    dist_df.plot(color=colors)
    plt.title("Total Distance Travelled by Iteration")
    plt.savefig(f"Images/{name}_total_dist.png") if name is not None else None
    plt.show()


if __name__ == "__main__":

    # define simulation names to be analyzed
    sim_name = "australia3"
    prefix = f"Data/{sim_name}"
    algorithms = ["todescato", "choi"]
    fidelities = ["hmf", "hsf", "nsf"]
    names = [f"{prefix}_{a}_{f}" for a in algorithms for f in fidelities]
    titles = [f"{a}_{f}" for a in algorithms for f in fidelities]

    # define dtypes to be loaded
    loss_dtypes = {"SimNum": int, "Iteration": int, "Period": int,
                   "Fidelity": "str", "Loss": float}
    agent_dtypes = {"SimNum": int, "Iteration": int, "Period": int,
                    "Fidelity": "str", "Agent": int,
                    "X": float, "Y": float, "XMax": float, "YMax": float,
                    "VarMax": float, "Var0": float,
                    "XCentroid": float, "YCentroid": float, "ProbExplore": float,
                    "Explore": object}

    # initialize dicts which will hold dataframes
    agents, losses, samples = {}, {}, {}
    for name, title in zip(names, titles):

        # read csvs in for each simulation
        agent_fname = f"{name}_agent.csv"
        agent_df = pd.read_csv(agent_fname, index_col=0, dtype=agent_dtypes)
        loss_fname = f"{name}_loss.csv"
        loss_df = pd.read_csv(loss_fname, index_col=0, dtype=loss_dtypes)
        # sample_fname = f"{name}_sample.csv"
        # sample_df = pd.read_csv(sample_fname, index_col=0)

        # add each df to its respective dictionary for easy analysis
        agents[title] = agent_df
        losses[title] = loss_df
        # samples[title] = sample_df

    # compute dists in place
    compute_dist(agents)

    # plot analysis
    plot_loss(losses, sim_name)
    plot_regret(losses, sim_name)
    plot_var(agents, sim_name)
    plot_explore(agents, sim_name)
    plot_dist(agents, sim_name)
    plot_total_dist(agents, sim_name)

