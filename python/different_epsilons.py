# The goal of this script is to run multiple METROPOLIS2 simulation and compare the results.
import os
import json

import numpy as np

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/different_epsilons/"

# Number of runs to simulate.
NB_RUNS = 10
# Number of agents to simulate.
N = 100_000
# Scale of the departure-time epsilons.
MU = 1.0
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
# Parameter of the learning model: weight of the current iteration relative to the previous value.
LEARNING_VALUE = 0.5
#  LEARNING_VALUE = "Best"
# Number of iterations to run.
NB_ITERATIONS = 200
# Interval of time, in seconds, between two breakpoints for the travel-time function.
RECORDING_INTERVAL = 60.0
# If `True`, analyze the results without running the simulation (only works if the simulation was
# already run before).
SKIP_RUN = False


if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    if not SKIP_RUN:
        for i in range(NB_RUNS):
            print(f"\n=====\nRun {i + 1}\n=====\n")
            directory = os.path.join(RUN_DIR, f"{i}")
            if not os.path.isdir(directory):
                os.makedirs(directory)

            print("Writing agents")
            agents = functions.get_agents(N, departure_time_mu=MU, random_seed=13081996 + i)
            with open(os.path.join(directory, "agents.json"), "w") as f:
                f.write(json.dumps(agents))

            print("Writing road network")
            road_network = functions.get_road_network(bottleneck_flow=BOTTLENECK_FLOW)
            with open(os.path.join(directory, "road-network.json"), "w") as f:
                f.write(json.dumps(road_network))

            print("Writing parameters")
            parameters = functions.get_parameters(
                learning_value=LEARNING_VALUE,
                nb_iteration=NB_ITERATIONS,
                recording_interval=RECORDING_INTERVAL,
            )
            with open(os.path.join(directory, "parameters.json"), "w") as f:
                f.write(json.dumps(parameters))

            print("Running simulation")
            functions.run_simulation(directory)

    print("Computing theoretical results")
    parameters = {
        "n": N,
        "mu": MU,
        "bottleneck_flow": BOTTLENECK_FLOW,
        "alpha": functions.ALPHA / 3600.0,
        "beta": functions.BETA / 3600.0,
        "gamma": functions.GAMMA / 3600.0,
        "tstar": functions.TSTAR,
        "delta": functions.DELTA,
        "period": functions.PERIOD,
        "tt0": functions.TT0,
    }
    times, denominator = theoretical_solution.equilibrium(parameters)

    print("Reading simulation results")
    running_times = list()
    for i in range(NB_RUNS):
        directory = os.path.join(RUN_DIR, f"{i}")
        running_times.append(functions.read_running_time(directory))
    print("Minimum running time: {:.2f} s".format(min(running_times)))
    print("Maximum running time: {:.2f} s".format(max(running_times)))
    print("Average running time: {:.2f} s".format(np.mean(running_times)))

    exp_tt_diff_rmses = list()
    exp_weights_rmse = list()
    dep_time_rmse = list()
    travel_times = list()
    distances = list()
    dep_times_list = list()
    for i in range(NB_RUNS):
        directory = os.path.join(RUN_DIR, f"{i}")
        iter_df = functions.read_iteration_results(directory)
        exp_tt_diff_rmses.append(iter_df["road_leg_exp_travel_time_diff_rmse"][-1])
        exp_weights_rmse.append(iter_df["exp_road_network_weights_rmse"][-1])
        dep_time_rmse.append(iter_df["trip_dep_time_rmse"][-1])
        travel_times.append(iter_df["road_leg_travel_time_mean"][-1])
        df = functions.read_leg_results(directory)
        distances.append(
            theoretical_solution.distance_theoretical(df, times, denominator, parameters)
        )
        dep_times_list.append(df["departure_time"].to_numpy())
    print("Minimum exp. travel time diff RMSE: {:.1E}".format(min(exp_tt_diff_rmses)))
    print("Maximum exp. travel time diff RMSE: {:.1E}".format(max(exp_tt_diff_rmses)))
    print("Average exp. travel time diff RMSE: {:.1E}".format(np.mean(exp_tt_diff_rmses)))
    print("Minimum exp. weights RMSE: {:.1E}".format(min(exp_weights_rmse)))
    print("Maximum exp. weights RMSE: {:.1E}".format(max(exp_weights_rmse)))
    print("Average exp. weights RMSE: {:.1E}".format(np.mean(exp_weights_rmse)))
    print("Minimum dep. time RMSE: {:.1E}".format(min(dep_time_rmse)))
    print("Maximum dep. time RMSE: {:.1E}".format(max(dep_time_rmse)))
    print("Average dep. time RMSE: {:.1E}".format(np.mean(dep_time_rmse)))
    print("Minimum average travel time: {:.4f} s".format(min(travel_times)))
    print("Maximum average travel time: {:.4f} s".format(max(travel_times)))
    print("Average average travel time: {:.4f} s".format(np.mean(travel_times)))
    print("Minimum theoretical distance: {:.4%}".format(min(distances)))
    print("Maximum theoretical distance: {:.4%}".format(max(distances)))
    print("Average theoretical distance: {:.4%}".format(np.mean(distances)))

    all_dep_times = np.concatenate(dep_times_list)
    all_dep_times.sort()
    n = len(all_dep_times)
    D = 0.0
    for i, dt in enumerate(all_dep_times):
        D = max(
            D,
            abs(i / NB_RUNS - theoretical_solution.integral(dt, times, denominator, parameters))
            / N,
        )
    print("Combined theoretical distance: {:.4%}".format(D))

    print("Plotting graphs")
    # Departure rate.
    bins = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 300)
    ts = (bins[1:] + bins[:-1]) / 2
    theoretical_rs = [theoretical_solution.dep_rate(t, denominator, times, parameters) for t in ts]
    simulated_rs, _ = np.histogram(all_dep_times, bins=bins, density=True)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for _, t in times.items():
        ax.plot(
            [t, t],
            [0, theoretical_solution.dep_rate(t, denominator, times, parameters)],
            color="black",
            linestyle="dashed",
        )
    ax.plot(
        ts,
        N * simulated_rs,
        color=mpl_utils.CMP(1),
        alpha=0.7,
        label="Simulated",
    )
    ax.plot(
        ts,
        theoretical_rs,
        linestyle="dashed",
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label="Analytical",
    )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel(r"Rate of departures from origin $r^{\text{d}}(t)$")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + list(times.values())
    time_labels = theoretical_solution.get_time_labels(times)
    labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + time_labels
    ax.set_xticks(all_times, labels=labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "different_epsilons_dep_rate.pdf"))
