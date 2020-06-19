"""
analysis.py

Scripts to be used in post-processing of log files. Generates visualizations and computes various metrics to
evaluate algorithm performance.

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/17/2020
"""

import pandas as pd
import matplotlib.pyplot as plt
import string


def compute_dist(agents):
    """

    :param agents:
    :return:
    """
    for title, df in agents.items():

        # compute difference from last iteration, fixing sim and agent
        diffs = df.groupby(by=["SimNum", "Agent"]).diff()

        # compute distance traveled as dx^2 + dy^2
        df["Distance"] = diffs["X"] ** 2 + diffs["Y"] ** 2
        df = df.fillna(0)


def plot_loss(losses):
    """

    :param losses:
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
    loss_df.plot()
    plt.title("Loss by Iteration")
    plt.show()

    # compute rolling loss and plot
    window_size = 10
    rolling_loss_df = loss_df.rolling(window_size, min_periods=1, center=True).mean()
    plt.figure()
    rolling_loss_df.plot()
    plt.title("Moving Average Loss by Iteration (n=10)")


def plot_regret(losses):
    """

    :param losses:
    :return:
    """
    regret_df = None
    for title, df in losses.items():

        # compute average of loss over iterations
        mean_loss = df.groupby(by="Iteration").mean()
        mean_loss = pd.DataFrame(mean_loss["Loss"])
        mean_loss.columns = [title]

        # subtract min value from all entries
        norm_loss = mean_loss - mean_loss.min()

        # cumulative-sum normalized loss to obtain regret
        regret = norm_loss.cumsum()

        # concatenate onto df of all simulation regrets (except first iteration)
        if regret_df is not None:
            regret_df = pd.concat([regret_df, regret], axis=1)
        else:
            regret_df = regret

    # plot df of all simulation regrets
    plt.figure()
    regret_df.plot()
    plt.title("Regret by Iteration")
    plt.show()
    #
    # # compute rolling loss and plot
    # window_size = 10
    # rolling_loss_df = loss_df.rolling(window_size, min_periods=1, center=True).mean()
    # plt.figure()
    # rolling_loss_df.plot()
    # plt.title("Moving Average Loss by Iteration (n=10)")


def plot_var(agents):
    """

    :param agents:
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
    var_df.plot()
    plt.title("Max Variance by Iteration")
    plt.show()


def plot_explore(agents):
    """

    :param agents:
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
    explore_df.plot()
    plt.title("Probability of Exploration by Iteration")
    plt.show()


def plot_dist(agents):
    """

    :param agents:
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
    dist_df.plot()
    plt.title("Distance Travelled by Iteration")
    plt.show()


def plot_total_dist(agents):
    """

    :param agents:
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
    dist_df.plot()
    plt.title("Total Distance Travelled by Iteration")
    plt.show()



if __name__ == "__main__":

    # define simulation names to be analyzed
    prefix = "Data/tc248"
    algorithms = ["todescato", "choi"]
    fidelities = ["nsf", "hsf", "hmf"]
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

    # plot items
    plot_loss(losses)
    plot_regret(losses)
    plot_var(agents)
    plot_explore(agents)
    plot_dist(agents)
    plot_total_dist(agents)
