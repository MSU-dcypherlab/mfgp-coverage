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
import time
import random
import cProfile
from multiprocessing import Pool
from simulator import todescato, choi

import numpy as np
import pandas as pd

""" Delimiter used in console output """
line_break = "\n" + "".join(["*" for i in range(100)]) + "\n"


def run_sim(args):

    (algo, sim_num, iterations, agents, truth, prior, hyp, console, plotter, log) = args

    print(line_break + f"Simulation {sim_num} : {algo}" + line_break)

    x_positions = [random.random() for i in range(agents)]
    y_positions = [random.random() for i in range(agents)]
    positions = np.column_stack((x_positions, y_positions))

    # 4) run simulation
    if "choi" in algo:
        loss_log_t, agent_log_t, sample_log_t = \
            choi(algo, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)
    else:
        loss_log_t, agent_log_t, sample_log_t = \
            todescato(algo, sim_num, iterations, agents, positions, truth, prior, hyp, console, plotter, log)

    return loss_log_t, agent_log_t, sample_log_t


def simulate_multiproc():
    """
    Run a series of multiagent learning-coverage algorithm simulations using multiprocessing
    """

    name = "Data/australia3"  # name of simulation, used as prefix of all associated input filenames
    prefix = "Data/australia3_multiproc"  # name of simulation, used as prefix of all associated output filenames

    agents = 4  # number of agents to use in simulation
    iterations = 120  # number of iterations to run each simulation
    simulations = 4  # number of simulations to run
    console = False  # boolean indicating if intermediate output should print to console
    log = True  # boolean indicating if output should be logged to CSV for performance analysis
    # plotter = Plotter([-eps, 1 + eps, -eps, 1 + eps])   # x_min, x_max, y_min, y_max
    plotter = None  # do not plot
    np.random.seed(1234)  # seed random generator for reproducibility
    n_processors = 4  # optimize for quad-core computer

    truth = pd.read_csv(f"{name}_hifi.csv")  # CSV specifying ground truth (x,y,z=f(x,y)) triples
    mf_hyp = pd.read_csv(f"{name}_mf_hyp.csv")  # CSV specifying multi-fidelity GP hyperparameters
    sf_hyp = pd.read_csv(f"{name}_sf_hyp.csv")  # CSV specifying single-fidelity GP hyperparameters
    null_prior = pd.read_csv("Data/null_prior.csv")  # Use a null prior
    human_prior = pd.read_csv(f"{name}_prior.csv")  # CSV specifying prior to condition GP upon before simulation

    algorithms = ["todescato_nsf", "choi_nsf"]

    for algo in algorithms:

        out_name = f"{prefix}_{algo}"
        loss_log, agent_log, sample_log = [], [], []  # reset logging lists for this algo

        # 1) select hyperparameters
        if "mf" in algo:
            hyp = mf_hyp
        else:
            hyp = sf_hyp

        # 2) select prior
        if "_n" in algo:
            prior = null_prior
        else:
            prior = human_prior

        # 3) configure arguments to pass to simulation
        args = [(algo, sim_num, iterations, agents, truth, prior, hyp, console, plotter, log)
                for sim_num in range(simulations)]

        # 4) pool and map simulations on all processors
        with Pool(processes=n_processors) as pool:
            out = pool.map(run_sim, args)

        # 5) reconstruct return data from multiprocessing
        for sim_num in range(simulations):
            loss_log.extend(out[sim_num][0])  # 0th element in each tuple is loss_log_t
            agent_log.extend(out[sim_num][1])  # 1st element in each tuple is agent_log_t
            sample_log.extend(out[sim_num][2])  # 2nd element in each tuple is sample_log_t

        # 6) save dataframes from simulation results for post-analysis
        if log:
            loss_df = pd.DataFrame(loss_log)
            loss_df.to_csv(out_name + "_loss.csv")
            agent_df = pd.DataFrame(agent_log)
            agent_df.to_csv(out_name + "_agent.csv")
            sample_df = pd.DataFrame(sample_log)
            sample_df.to_csv(out_name + "_sample.csv")


def simulate():
    """
    Run a series of multiagent learning-coverage algorithm simulations.
    """

    name = "Data/australia3"           # name of simulation, used as prefix of all associated input filenames
    prefix = "Data/australia3_single"     # name of simulation, used as prefix of all associated output filenames

    agents = 4              # number of agents to use in simulation
    iterations = 120       # number of iterations to run each simulation
    simulations = 4        # number of simulations to run
    console = False          # boolean indicating if intermediate output should print to console
    log = True              # boolean indicating if output should be logged to CSV for performance analysis
    # plotter = Plotter([-eps, 1 + eps, -eps, 1 + eps])   # x_min, x_max, y_min, y_max
    plotter = None          # do not plot
    np.random.seed(1234)    # seed random generator for reproducibility

    truth = pd.read_csv(f"{name}_hifi.csv")         # CSV specifying ground truth (x,y,z=f(x,y)) triples
    mf_hyp = pd.read_csv(f"{name}_mf_hyp.csv")      # CSV specifying multi-fidelity GP hyperparameters
    sf_hyp = pd.read_csv(f"{name}_sf_hyp.csv")      # CSV specifying single-fidelity GP hyperparameters
    null_prior = pd.read_csv("Data/null_prior.csv")      # Use a null prior
    human_prior = pd.read_csv(f"{name}_prior.csv")        # CSV specifying prior to condition GP upon before simulation

    algorithms = ["todescato_nsf", "choi_nsf"]


    for algo in algorithms:

        out_name = f"{prefix}_{algo}"
        loss_log, agent_log, sample_log = [], [], []  # reset logging lists for this algo

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


if __name__ == "__main__":

    start = time.time()
    simulate_multiproc()
    end = time.time()
    print(f"\nMultiproc total time: {(end - start)}")

    start = time.time()
    simulate()
    end = time.time()
    print(f"\nStandard total time: {(end - start)}")