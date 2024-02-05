# How METROPOLIS2 convergence varies with the learning model parameter.
import os
import json

import numpy as np

import functions
import mpl_utils
import theoretical_solution

SUFFIX = ""
# Path to the directory where the simulation input should be stored.
RUN_DIR = f"./runs/convergence_learning_model{SUFFIX}/"
# Number of agents to simulate.
N = 100_000
# Values to test for the parameter of the learning model.
PARAMETERS = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0]
# Number of iterations to run.
NB_ITERATIONS = 1000
# Value of mu for the departure-time choice model.
MU = 1.0
# Interval of time, in seconds, between two breakpoints for the travel-time function.
RECORDING_INTERVAL = 60.0
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
# If `True`, analyze the results without running the simulations (only works if the simulations
# where already run before).
SKIP_RUN = False


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
                learning_value=parameter,
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

    print("Reading results")
    iteration_results = list()
    for i, parameter in enumerate(PARAMETERS):
        print("===== λ = {} =====".format(parameter))
        directory = os.path.join(RUN_DIR, f"{i}")

        print("Running time: {:.2f} s".format(functions.read_running_time(directory)))

        df = functions.read_leg_results(directory)
        D = theoretical_solution.distance_theoretical(df, times, denominator, parameters)
        print("D = {:.4%}".format(D))

        df = functions.read_iteration_results(directory)
        iteration_results.append(df)
        print("Exp. travel time RMSE: {:.2E}".format(df["road_leg_exp_travel_time_diff_rmse"][-1]))
        print("Sim weight RMSE: {:.2E}".format(df["sim_road_network_weights_rmse"][-1]))
        print("Dep. time RMSE: {:.2E}".format(df["trip_dep_time_rmse"][-1]))

        print("Average travel time: {:.4f}s".format(df["road_leg_travel_time_mean"][-1]))
        print(
            "Average congested travel time: {:.4f}s".format(
                df["road_leg_travel_time_mean"][-1] - df["road_leg_road_time_mean"][-1]
            )
        )

    # Departure-time RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["trip_dep_time_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\lambda = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{dep}}$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(mpl_utils.GRAPH_DIR, f"convergence_learning_model_dep_time_rmse{SUFFIX}.pdf")
    )

    # Expected travel time diff. RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["road_leg_exp_travel_time_diff_rmse"],
            alpha=0.7,
            color=mpl_utils.CMP(i),
            label=r"$\lambda = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{tt}}$ (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            mpl_utils.GRAPH_DIR, f"convergence_learning_model_exp_travel_time_diff_rmse{SUFFIX}.pdf"
        )
    )

    # Expected travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["exp_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\lambda = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel("Expected travel-time function RMSE (seconds)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            mpl_utils.GRAPH_DIR, f"convergence_learning_model_exp_weights_rmse{SUFFIX}.pdf"
        )
    )

    # Simulated travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["sim_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\lambda = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^T$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            mpl_utils.GRAPH_DIR, f"convergence_learning_model_sim_weights_rmse{SUFFIX}.pdf"
        )
    )

    # Departure rate.
    bins = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 300)
    ts = (bins[1:] + bins[:-1]) / 2
    theoretical_rs = [theoretical_solution.dep_rate(t, denominator, times, parameters) for t in ts]
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
        theoretical_rs,
        linestyle="dashed",
        color="black",
        alpha=0.7,
        label="Analytical",
    )
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        directory = os.path.join(RUN_DIR, f"{i}")
        df = functions.read_leg_results(directory)
        simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
        ax.plot(
            ts,
            N * simulated_rs,
            color=mpl_utils.CMP(i),
            alpha=0.7,
            label=r"$\lambda = {}$".format(parameter),
        )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel(r"Rate of departures from origin $r^{\text{d}}(t)$")
    ax.set_ylim(bottom=0)
    ax.legend()
    all_times = functions.PERIOD + list(times.values())
    time_labels = theoretical_solution.get_time_labels(times)
    labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + time_labels
    ax.set_xticks(all_times, labels=labels)
    fig.tight_layout()
    fig.savefig(
        os.path.join(mpl_utils.GRAPH_DIR, f"convergence_learning_model_dep_rate{SUFFIX}.pdf")
    )
