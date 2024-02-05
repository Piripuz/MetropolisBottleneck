# How METROPOLIS2 convergence varies with the breakpoint interval.
import os
import json

import numpy as np

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/impact_breakpoint_interval/"
# Values to test for the breakpoint interval δ.
PARAMETERS = [60.0, 300.0, 600.0, 900.0]
# Number of agents to simulate.
N = 100_000
# Parameter of the learning model.
LEARNING_VALUE = 0.5
# Number of iterations to run.
NB_ITERATIONS = 200
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
# Value of mu for the departure-time choice model.
MU = 1.0
# If `True`, analyze the results without running the simulations (only works if the simulations
# where already run before).
SKIP_RUN = False


def seconds_to_travel_time_str(t):
    m = int(t // 60)
    s = int(round(t % 60))
    string = ""
    if m > 0:
        string += rf"{m}\:min"
    if s > 0:
        string += rf"{s}\:sec"
    return string


if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    if not SKIP_RUN:
        for i, parameter in enumerate(PARAMETERS):
            print(f"\n=====\nNew parameter value: {parameter}\n=====\n")
            directory = os.path.join(RUN_DIR, f"{i}")
            if not os.path.isdir(directory):
                os.makedirs(directory)

            print("Writing input")
            agents = functions.get_agents(N, departure_time_mu=MU)
            with open(os.path.join(directory, "agents.json"), "w") as f:
                f.write(json.dumps(agents))
            road_network = functions.get_road_network(bottleneck_flow=BOTTLENECK_FLOW)
            with open(os.path.join(directory, "road-network.json"), "w") as f:
                f.write(json.dumps(road_network))
            parameters = functions.get_parameters(
                learning_value=LEARNING_VALUE,
                nb_iteration=NB_ITERATIONS,
                recording_interval=parameter,
            )
            with open(os.path.join(directory, "parameters.json"), "w") as f:
                f.write(json.dumps(parameters))

            print("Running simulation")
            functions.run_simulation(directory)

    print("Computing theoretical results")
    parameters = {
        "n": N,
        "bottleneck_flow": BOTTLENECK_FLOW,
        "mu": MU,
        "alpha": functions.ALPHA / 3600.0,
        "beta": functions.BETA / 3600.0,
        "gamma": functions.GAMMA / 3600.0,
        "tstar": functions.TSTAR,
        "delta": functions.DELTA,
        "period": functions.PERIOD,
        "tt0": functions.TT0,
    }
    times, denominator = theoretical_solution.equilibrium(parameters)

    print("Reading results")
    iteration_results = list()
    for i, parameter in enumerate(PARAMETERS):
        print("===== δ = {:} =====".format(parameter))
        directory = os.path.join(RUN_DIR, f"{i}")

        print("Running time: {:.2f} s".format(functions.read_running_time(directory)))

        df = functions.read_leg_results(directory)
        D = theoretical_solution.distance_theoretical(df, times, denominator, parameters)
        print("D = {:.4%}".format(D))

        df = functions.read_iteration_results(directory)
        iteration_results.append(df)
        print(
            "Exp. travel time diff RMSE: {:.1E}".format(
                df["road_leg_exp_travel_time_diff_rmse"][-1]
            )
        )
        print("Exp weight RMSE: {:.1E}".format(df["exp_road_network_weights_rmse"][-1]))
        print("Dep. time RMSE: {:.1E}".format(df["trip_dep_time_rmse"][-1]))

        print("Average travel time: {:.4f}s".format(df["road_leg_travel_time_mean"][-1]))

    # Departure-time RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["trip_dep_time_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\delta =$\:{}".format(seconds_to_travel_time_str(parameter)),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{dep}}$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_interval_dep_time_rmse.pdf"))

    # Expected travel time diff. RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["road_leg_exp_travel_time_diff_rmse"],
            alpha=0.7,
            color=mpl_utils.CMP(i),
            label=r"$\delta =$\:{}".format(seconds_to_travel_time_str(parameter)),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{tt}}$ (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_interval_exp_travel_time_diff_rmse.pdf"))

    # Expected travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["exp_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\delta =$\:{}".format(seconds_to_travel_time_str(parameter)),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^T$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_interval_exp_weights_rmse.pdf"))

    # Simulated travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["sim_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\delta =$\:{}".format(seconds_to_travel_time_str(parameter)),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel("Simulated travel-time function RMSE (seconds)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_interval_sim_weights_rmse.pdf"))

    # Travel-time function.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for _, t in times.items():
        ax.plot(
            [t, t],
            [0, theoretical_solution.travel_time(t, denominator, times, parameters)],
            color="black",
            linestyle="dashed",
        )
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        directory = os.path.join(RUN_DIR, f"{i}")
        weights = functions.read_sim_weight_results(directory)
        simulated_tt = weights["points"]
        ts = np.arange(
            functions.PERIOD[0],
            functions.PERIOD[0] + len(weights["points"]) * weights["interval_x"],
            weights["interval_x"],
        )
        ax.plot(
            ts,
            weights["points"],
            "-o",
            markersize=3,
            color=mpl_utils.CMP(1 + i),
            alpha=0.7,
            label=r"$\delta =$\:{}".format(seconds_to_travel_time_str(parameter)),
        )
    #  ts = np.arange(
    #  functions.PERIOD[0],
    #  functions.PERIOD[1] + 1.0,
    #  1.0,
    #  )
    #  tts = np.fromiter(
    #  (theoretical_solution.travel_time(t, denominator, times, parameters) for t in ts),
    #  dtype=np.float64,
    #  )
    #  ax.plot(
    #  ts,
    #  tts,
    #  linestyle="dashed",
    #  color=mpl_utils.CMP(0),
    #  alpha=0.7,
    #  label="Analytical",
    #  )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Travel time $T(t)$ (seconds)")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + list(times.values())
    time_labels = theoretical_solution.get_time_labels(times)
    labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + time_labels
    ax.set_xticks(all_times, labels=labels)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    yticks = np.append(yticks, functions.TT0)
    yticklabels = np.append(yticklabels, "$t_f$")
    ax.set_yticks(yticks, labels=yticklabels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_interval_travel_time_function.pdf"))
