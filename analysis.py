"""
analysis.py

Scripts to be used in post-processing of log files. Generates visualizations and computes various metrics to
evaluate algorithm performance.

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/17/2020
"""

from math import sqrt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



todescato_colors = [(0, 120, 120),        # dark teal
                    (0, 180, 180),      # mid teal
                    (0, 255, 255)]      # bright teal
choi_colors = [(120, 0, 120),             # dark purple
               (180, 0, 180),           # mid purple
               (255, 0, 255)]           # bright purple
periodic_colors = [(120, 120, 0),         # dark yellow
                   (180, 180, 0),       # mid yellow
                   (255, 255, 0)]       # bright yellow
lloyd_color = [(180, 180, 180)]         # grey

base_colors = todescato_colors + choi_colors + periodic_colors + lloyd_color    # define colors in (0, 255)
colors = [tuple(map(lambda val: val/255, color)) for color in base_colors]      # normalize colors in (0, 1)


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

    Referenced https://stackoverflow.com/questions/45222084/plotting-fill-between-in-pandas 6/29/2020

    :param losses:
    :param name:
    :return:
    """
    mean_loss_df = None
    std_loss_df = None

    for title, df in losses.items():

        num_simulations = df["SimNum"].max() + 1    # starts counting at 0

        # compute average of loss over iterations
        mean_loss = df.groupby(by="Iteration").mean()
        mean_loss = pd.DataFrame(mean_loss["Loss"])
        mean_loss.columns = [title]

        # compute std dev of loss over iterations
        std_loss = df.groupby(by="Iteration").std()
        std_loss = pd.DataFrame(std_loss["Loss"])
        std_loss.columns = [title]

        # concatenate onto df of all simulation losses (except first iteration)
        if mean_loss_df is not None:
            mean_loss_df = pd.concat([mean_loss_df, mean_loss], axis=1)
            std_loss_df = pd.concat([std_loss_df, std_loss], axis=1)
        else:
            mean_loss_df = mean_loss
            std_loss_df = std_loss

    # plot df of all simulation losses
    plt.figure()
    ax = mean_loss_df.plot(color=colors)
    for i in range(len(mean_loss_df.columns)):
        ax.fill_between(x=mean_loss_df.index,
                        y1=mean_loss_df.iloc[:, i] - 2*std_loss_df.iloc[:, i] / sqrt(num_simulations),
                        y2=mean_loss_df.iloc[:, i] + 2*std_loss_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Loss by Iteration")
    plt.savefig(f"Images/{name}/{name}_loss.png") if name is not None else None
    plt.show()

    # plot closeup of simulation loss with periodic dropped out
    mean_loss_df = mean_loss_df.drop(columns=["periodic_nsf", "periodic_hsf", "periodic_hmf"])
    std_loss_df = std_loss_df.drop(columns=["periodic_nsf", "periodic_hsf", "periodic_hmf"])
    plt.figure()
    ax = mean_loss_df.plot(color=colors)
    for i in range(len(mean_loss_df.columns)):
        ax.fill_between(x=mean_loss_df.index,
                        y1=mean_loss_df.iloc[:, i] - 2 * std_loss_df.iloc[:, i] / sqrt(num_simulations),
                        y2=mean_loss_df.iloc[:, i] + 2 * std_loss_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Loss by Iteration: Zoomed")
    plt.ylim((0.005, 0.012))
    plt.savefig(f"Images/{name}/{name}_loss_zoomed.png") if name is not None else None
    plt.show()


    # compute rolling loss and plot
    # window_size = 10
    # rolling_mean_loss_df = mean_loss_df.rolling(window_size, min_periods=1, center=True).mean()
    # plt.figure()
    # rolling_mean_loss_df.plot(color=colors)
    # plt.title("Moving Average Loss by Iteration (n=10)")
    # plt.savefig(f"Images/{name}/{name}_loss_rolling.png") if name is not None else None
    # plt.show()


def plot_regret(losses, name=None):
    """

    Referenced https://stackoverflow.com/questions/45222084/plotting-fill-between-in-pandas 6/29/2020

    :param losses:
    :param name:
    :return:
    """
    regret_df = None
    std_regret_df = None

    min_loss = losses["lloyd"]["Loss"].min()
    # min_loss = [df["Loss"].min() for df in losses.values()]
    # min_loss = min(min_loss)
    for title, df in losses.items():

        num_simulations = df["SimNum"].max() + 1    # starts counting at 0

        # compute average of loss over iterations
        mean_loss = df.groupby(by="Iteration").mean()
        mean_loss = pd.DataFrame(mean_loss["Loss"])
        mean_loss.columns = [title]

        # subtract min value from all entries
        # last_loss = mean_loss.iloc[-1]
        # min_loss = mean_loss.min()
        norm_loss = np.maximum(mean_loss - min_loss, 0)

        # cumulative-sum normalized loss to obtain regret
        regret = norm_loss.cumsum()

        # compute std dev of loss over iterations
        std_loss = df.groupby(by="Iteration").std()
        std_loss = pd.DataFrame(std_loss["Loss"])
        std_loss.columns = [title]
        std_regret = std_loss.cumsum()

        # concatenate onto df of all simulation regrets (except first iteration)
        if regret_df is not None:
            regret_df = pd.concat([regret_df, regret], axis=1)
            std_regret_df = pd.concat([std_regret_df, std_regret], axis=1)
        else:
            regret_df = regret
            std_regret_df = std_regret

    # plot df of all simulation regrets
    plt.figure()
    ax = regret_df.plot(color=colors)
    for i in range(len(regret_df.columns)):
        ax.fill_between(x=regret_df.index,
                        y1=regret_df.iloc[:, i] - 2*std_regret_df.iloc[:, i] / sqrt(num_simulations),
                        y2=regret_df.iloc[:, i] + 2*std_regret_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Regret by Iteration")
    plt.savefig(f"Images/{name}/{name}_regret.png") if name is not None else None
    plt.show()

    # drop periodic and zoom in
    regret_df = regret_df.drop(columns=["periodic_nsf", "periodic_hsf", "periodic_hmf"])
    std_regret_df = std_regret_df.drop(columns=["periodic_nsf", "periodic_hsf", "periodic_hmf"])
    plt.figure()
    ax = regret_df.plot(color=colors)
    for i in range(len(regret_df.columns)):
        ax.fill_between(x=regret_df.index,
                        y1=regret_df.iloc[:, i] - 2 * std_regret_df.iloc[:, i] / sqrt(num_simulations),
                        y2=regret_df.iloc[:, i] + 2 * std_regret_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Regret by Iteration: Zoomed")
    plt.ylim((0, 0.5))
    plt.savefig(f"Images/{name}/{name}_regret_zoomed.png") if name is not None else None
    plt.show()


def plot_var(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    var_df = None
    std_var_df = None

    for title, df in agents.items():

        num_simulations = df["SimNum"].max() + 1    # starts counting at 0

        # compute average of posterior max var over iterations
        max_var_by_sim = df.groupby(by=["SimNum", "Iteration"]).max()
        mean_max_var = max_var_by_sim.groupby(by="Iteration").mean()
        mean_max_var = pd.DataFrame(mean_max_var["VarMax"])
        mean_max_var.columns = [title]

        # compute std of posterior max var over iterations
        std_max_var = max_var_by_sim.groupby(by="Iteration").std()
        std_max_var = pd.DataFrame(std_max_var["VarMax"])
        std_max_var.columns = [title]

        # concatenate onto df of all simulation vars (except first iteration)
        if var_df is not None:
            var_df = pd.concat([var_df, mean_max_var], axis=1)
            std_var_df = pd.concat([std_var_df, std_max_var], axis=1)
        else:
            var_df = mean_max_var
            std_var_df = std_max_var

    # plot df of all simulation vars
    plt.figure()
    ax = var_df.plot(color=colors)
    for i in range(len(var_df.columns)):
        ax.fill_between(x=var_df.index,
                        y1=var_df.iloc[:, i] - 2*std_var_df.iloc[:, i] / sqrt(num_simulations),
                        y2=var_df.iloc[:, i] + 2*std_var_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Max Variance by Iteration")
    plt.savefig(f"Images/{name}/{name}_var.png") if name is not None else None
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
    plt.savefig(f"Images/{name}/{name}_explore.png") if name is not None else None
    plt.show()



def plot_dist(agents, name=None):
    """

    :param agents:
    :param name:
    :return:
    """
    dist_df = None
    std_dist_df = None
    total_dist_df = None
    std_total_dist_df = None

    for title, df in agents.items():

        num_simulations = df["SimNum"].max() + 1    # starts counting at 0

        # sum distance traveled over all agents for fixed simnum, iteration
        distance_sum = df.groupby(by=["SimNum", "Iteration"]).sum()

        # average distance by all agents per iteration for fixed iteration
        distance = distance_sum.groupby(by="Iteration").mean()
        distance = pd.DataFrame(distance["Distance"])
        distance.columns = [title]

        # compute std dev of distance by iteration
        std_distance = distance_sum.groupby(by="Iteration").std()
        std_distance = pd.DataFrame(std_distance["Distance"])
        std_distance.columns = [title]

        # compute total distance as cumulative sum of distance per iteration
        total_distance = pd.DataFrame(distance.cumsum())
        total_distance.columns = [title]

        # compute std dev of total distance as cumulative sum
        std_total_distance = pd.DataFrame(std_distance.cumsum())
        std_total_distance.columns = [title]

        # concatenate onto df of all distances traveled (except first iteration)
        if dist_df is not None:
            dist_df = pd.concat([dist_df, distance], axis=1)
            total_dist_df = pd.concat([total_dist_df, total_distance], axis=1)
            std_dist_df = pd.concat([std_dist_df, std_distance], axis=1)
            std_total_dist_df = pd.concat([std_total_dist_df, std_total_distance], axis=1)
        else:
            dist_df = distance
            total_dist_df = total_distance
            std_dist_df = std_distance
            std_total_dist_df = std_total_distance

    # plot df of all distances traveled
    plt.figure()
    ax = dist_df.plot(color=colors)
    for i in range(len(dist_df.columns)):
        ax.fill_between(x=dist_df.index,
                        y1=dist_df.iloc[:, i] - 2*std_dist_df.iloc[:, i] / sqrt(num_simulations),
                        y2=dist_df.iloc[:, i] + 2*std_dist_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Distance Travelled by Iteration")
    plt.savefig(f"Images/{name}/{name}_dist.png") if name is not None else None
    plt.show()

    # plot df of all total distances traveled
    plt.figure()
    ax = total_dist_df.plot(color=colors)
    for i in range(len(total_dist_df.columns)):
        ax.fill_between(x=total_dist_df.index,
                        y1=total_dist_df.iloc[:, i] - 2*std_total_dist_df.iloc[:, i] / sqrt(num_simulations),
                        y2=total_dist_df.iloc[:, i] + 2*std_total_dist_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Total Distance Travelled by Iteration")
    plt.savefig(f"Images/{name}/{name}_total_dist.png") if name is not None else None
    plt.show()


def plot_samples(agents, name=None):
    """

    :param samples:
    :param name:
    :return:
    """
    sample_df = None
    std_sample_df = None
    total_sample_df = None
    std_total_sample_df = None

    for title, df in agents.items():

        num_simulations = df["SimNum"].max() + 1    # starts counting at 0

        # map explore T/F column to 0/1
        df["Explore"] = df["Explore"].apply(lambda x: int(eval(x)))

        # compute mean number of samples by iteration
        mean_samples = df.groupby("Iteration").mean()
        mean_samples = pd.DataFrame(mean_samples["Explore"])

        # compute std samples by iteration
        std_samples = df.groupby("Iteration").std()
        std_samples = pd.DataFrame(std_samples["Explore"])

        # relabel
        mean_samples.columns = [title]
        std_samples.columns = [title]

        # compute mean cumulative sum of number of samples by iteration
        total_samples = mean_samples.cumsum()
        std_total_samples = std_samples.cumsum()

        # concatenate onto df of all sample dfs(except first iteration)
        if sample_df is not None:
            sample_df = pd.concat([sample_df, mean_samples], axis=1)
            total_sample_df = pd.concat([total_sample_df, total_samples], axis=1)
            std_sample_df = pd.concat([std_sample_df, std_samples], axis=1)
            std_total_sample_df = pd.concat([std_total_sample_df, std_total_samples], axis=1)
        else:
            sample_df = mean_samples
            total_sample_df = total_samples
            std_sample_df = std_samples
            std_total_sample_df = std_total_samples

    # plot df of mean samples by iteration
    plt.figure()
    ax = sample_df.plot(color=colors)
    for i in range(len(sample_df.columns)):
        ax.fill_between(x=sample_df.index,
                        y1=sample_df.iloc[:, i] - 2*std_sample_df.iloc[:, i] / sqrt(num_simulations),
                        y2=sample_df.iloc[:, i] + 2*std_sample_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Mean # Samples by Iteration")
    plt.savefig(f"Images/{name}/{name}_samples.png") if name is not None else None
    plt.show()

    # plot df of mean cumulative sum of samples by iteration
    plt.figure()
    ax = total_sample_df.plot(color=colors)
    for i in range(len(total_sample_df.columns)):
        ax.fill_between(x=total_sample_df.index,
                        y1=total_sample_df.iloc[:, i] - 2*std_total_sample_df.iloc[:, i] / sqrt(num_simulations),
                        y2=total_sample_df.iloc[:, i] + 2*std_total_sample_df.iloc[:, i] / sqrt(num_simulations),
                        color=colors[i], alpha=0.2)
    plt.title("Mean Cumulative Sum of # Samples by Iteration")
    plt.savefig(f"Images/{name}/{name}_total_samples.png") if name is not None else None
    plt.show()


if __name__ == "__main__":

    # define simulation names to be analyzed
    sim_name = "australia7"
    prefix = f"Data/{sim_name}"
    algorithms = ["todescato", "choi", "periodic"]
    fidelities = ["nsf", "hsf", "hmf"]
    names = [f"{prefix}_{a}_{f}" for a in algorithms for f in fidelities]
    names.append(f"{prefix}_lloyd")
    titles = [f"{a}_{f}" for a in algorithms for f in fidelities]
    titles.append("lloyd")

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
        sample_fname = f"{name}_sample.csv"
        sample_df = pd.read_csv(sample_fname, index_col=0)

        # add each df to its respective dictionary for easy analysis
        agents[title] = agent_df
        losses[title] = loss_df
        samples[title] = sample_df

    # compute dists in place
    # compute_dist(agents)

    # plot analysis
    # plot_loss(losses, sim_name)
    plot_regret(losses, sim_name)
    # plot_var(agents, sim_name)
    # plot_explore(agents, sim_name)
    # plot_dist(agents, sim_name)
    # plot_samples(agents, sim_name)
