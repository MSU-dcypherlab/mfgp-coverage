"""
runner.py

Efficiently runs repeated simulations using multiprocessing for mean performance comparison

by: Andrew McDonald, D-CYPHER Lab, Michigan State University
last modified: 6/11/2020
"""

import sys
import copy
import time
import random
import cProfile
from multiprocessing import Pool
from simulator import todescato, choi, lloyd, periodic
from plotter import Plotter

import numpy as np
import pandas as pd


""" Boundary cushion to be used in computations with and plots of unit square """
eps = 0.1

""" Delimiter used in console output """
line_break = "\n" + "".join(["-" for i in range(100)]) + "\n"

""" Delimiter used in console output """
slash_break = "\n" + "".join(["/" for i in range(100)]) + "\n"


def run_sim(args):

    # 1) unpack parameters and start timer
    (out_name, algo, sim_num, iterations, agents, truth, sigma_n, prior, hyp, console, plotter, log) = args
    print(line_break + f"Start Simulation {sim_num} : {algo}" + line_break)
    sim_start = time.time()

    # 2) randomly generate starting positions of agents
    x_positions = [random.random() for i in range(agents)]
    y_positions = [random.random() for i in range(agents)]
    positions = np.column_stack((x_positions, y_positions))

    # 3) select and run proper algorithm
    if "choi" in algo:
        loss_log_t, agent_log_t, sample_log_t = \
            choi(algo, sim_num, iterations, agents, positions, truth, sigma_n, prior, hyp, console, plotter, log)
    elif "todescato" in algo:
        loss_log_t, agent_log_t, sample_log_t = \
            todescato(algo, sim_num, iterations, agents, positions, truth, sigma_n, prior, hyp, console, plotter, log)
    elif "lloyd" in algo:
        loss_log_t, agent_log_t, sample_log_t = \
            lloyd(algo, sim_num, iterations, agents, positions, truth, sigma_n, prior, hyp, console, plotter, log)
    elif "periodic" in algo:
        loss_log_t, agent_log_t, sample_log_t = \
            periodic(algo, sim_num, iterations, agents, positions, truth, sigma_n, prior, hyp, console, plotter, log)
    else:
        raise ValueError("Invalid simulation algorithm specified.")

    # 4) save ending configuration if plotter is enabled
    plotter.save(f"{out_name}.png") if plotter else None

    # 5) end time and return data
    sim_end = time.time()
    print(line_break + f"End Simulation {sim_num} : {algo}\n"
                       f"Time : {sim_end - sim_start}" + line_break)

    return loss_log_t, agent_log_t, sample_log_t


def run(n_processors=4):
    """
    Run a series of multiagent learning-coverage algorithm simulations using multiprocessing

    :param n_processors: [int] number of processors over which to distribute computation (set equal to CPU cores)
    """

    # 1) define simulation hyperparameters
    name = "Data/australia9"  # name of simulation, used as prefix of all associated input filenames
    prefix = "Data/australia9.1"  # name of simulation, used as prefix of all associated output filenames

    agents = 8          # number of agents to use in simulation
    iterations = 120    # number of iterations to run each simulation
    simulations = 100     # number of simulations to run
    sigma_n = 0.1       # sampling noise std. dev. on hifi data (should match distribution's generational parameter)
    console = False      # boolean indicating if intermediate output should print to console
    log = True          # boolean indicating if output should be logged to CSV for performance analysis
    # plotter = Plotter([-eps, 1 + eps, -eps, 1 + eps])   # x_min, x_max, y_min, y_max
    plotter = None      # do not plot
    np.random.seed(1234)  # seed random generator for reproducibility

    algorithms = ["todescato_nsf",
                  "choi_nsf",
                  "todescato_hsf",
                  "choi_hsf",
                  "todescato_hmf",
                  "choi_hmf",
                  # "periodic_nsf", "periodic_hsf", "periodic_hmf",
                  "lloyd"]

    # 2) load distributional data
    truth = pd.read_csv(f"{name}_hifi.csv")  # CSV specifying ground truth (x,y,z=f(x,y)) triples
    mf_hyp = pd.read_csv(f"{name}_mf_hyp.csv")  # CSV specifying multi-fidelity GP hyperparameters
    sf_hyp = pd.read_csv(f"{name}_sf_hyp.csv")  # CSV specifying single-fidelity GP hyperparameters
    null_prior = pd.read_csv("Data/null_prior.csv")  # Use a null prior
    human_prior = pd.read_csv(f"{name}_prior.csv")  # CSV specifying prior to condition GP upon before simulation

    # 3) run each algorithm sequentially, repeating "simulations" times with multiprocessing
    for algo in algorithms:

        print(slash_break + f"Start Algorithm : {algo}" + slash_break)
        algo_start = time.time()

        out_name = f"{prefix}_{algo}"
        loss_log, agent_log, sample_log = [], [], []  # reset logging lists for this algo

        # 4) select hyperparameters for this algorithm
        if "mf" in algo:
            hyp = mf_hyp
        else:
            hyp = sf_hyp

        # 5) select prior for this algorithm
        if "_n" in algo:
            prior = null_prior
        else:
            prior = human_prior

        # 6) configure arguments to pass to simulation
        args = [(out_name, algo, sim_num, iterations, agents, truth, sigma_n, prior, hyp, console, plotter, log)
                for sim_num in range(simulations)]

        # 7) pool and map simulations on all processors
        if n_processors > 1:
            with Pool(processes=n_processors) as pool:
                out = pool.map(run_sim, args)
        else:
            out = []
            for arg in args:
                out.append(run_sim(arg))

        # 8) reconstruct return data from multiprocessing
        for sim_num in range(simulations):
            loss_log.extend(out[sim_num][0])  # 0th element in each tuple is loss_log_t
            agent_log.extend(out[sim_num][1])  # 1st element in each tuple is agent_log_t
            sample_log.extend(out[sim_num][2])  # 2nd element in each tuple is sample_log_t

        # 9) save dataframes from simulation results for post-analysis
        if log:
            loss_df = pd.DataFrame(loss_log)
            loss_df.to_csv(f"{out_name}_loss.csv")
            agent_df = pd.DataFrame(agent_log)
            agent_df.to_csv(f"{out_name}_agent.csv")
            sample_df = pd.DataFrame(sample_log)
            sample_df.to_csv(f"{out_name}_sample.csv")

        algo_end = time.time()
        print(slash_break + f"End Algorithm : {algo}\n"
                            f"Time : {algo_end - algo_start}\n"
                            f"Time/Sim : {(algo_end - algo_start) / simulations}" + slash_break)


if __name__ == "__main__":

    start = time.time()
    run(n_processors=4)
    end = time.time()
    print(slash_break + slash_break +
          f"runner.py Total Time : {end - start}\n" +
          slash_break + slash_break)
