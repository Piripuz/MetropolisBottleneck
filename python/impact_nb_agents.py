# How METROPOLIS2 convergence varies with the number of agents simulated.
import os
import json

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/impact_nb_agents/"
# Values to test for the number of agents to simulate.
PARAMETERS = [10, 100, 1_000, 10_000, 100_000]
# Parameter of the learning model.
LEARNING_VALUE = 0.5
# Number of iterations to run.
NB_ITERATIONS = 200
# Value of mu for the departure-time choice model.
MU = 1.0
# Interval of time, in seconds, between two breakpoints for the travel-time function.
RECORDING_INTERVAL = 60.0
# If `True`, analyze the results without running the simulations (only works if the simulations
# where already run before).
SKIP_RUN = False

if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    if not SKIP_RUN:
        for i, parameter in enumerate(PARAMETERS):
            print(f"\n=====\nNew parameter value: {parameter:,}\n=====\n")
            directory = os.path.join(RUN_DIR, f"{i}")
            if not os.path.isdir(directory):
                os.makedirs(directory)

            bottleneck_flow = parameter / (3600.0 * 2 / 3)

            print("Writing input")
            agents = functions.get_agents(parameter, departure_time_mu=MU)
            with open(os.path.join(directory, "agents.json"), "w") as f:
                f.write(json.dumps(agents))
            road_network = functions.get_road_network(bottleneck_flow=bottleneck_flow)
            with open(os.path.join(directory, "road-network.json"), "w") as f:
                f.write(json.dumps(road_network))
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
        "mu": MU,
        "alpha": functions.ALPHA / 3600.0,
        "beta": functions.BETA / 3600.0,
        "gamma": functions.GAMMA / 3600.0,
        "tstar": functions.TSTAR,
        "delta": functions.DELTA,
        "period": functions.PERIOD,
        "tt0": functions.TT0,
    }

    print("Reading results")
    iteration_results = list()
    for i, parameter in enumerate(PARAMETERS):
        print("===== N = {:,} =====".format(parameter))
        directory = os.path.join(RUN_DIR, f"{i}")

        print("Running time: {:.2f} s".format(functions.read_running_time(directory)))

        # Theoretical results.
        parameters["n"] = parameter
        parameters["bottleneck_flow"] = parameter / (3600.0 * 2 / 3)
        times, denominator = theoretical_solution.equilibrium(parameters)

        df = functions.read_leg_results(directory)
        D = theoretical_solution.distance_theoretical(df, times, denominator, parameters)
        print("D = {:.4%}".format(D))

        df = functions.read_iteration_results(directory)
        iteration_results.append(df)
        print("Exp. travel time RMSE: {:.1E}".format(df["road_leg_exp_travel_time_diff_rmse"][-1]))
        print("Sim weight RMSE: {:.1E}".format(df["sim_road_network_weights_rmse"][-1]))
        print("Dep. time RMSE: {:.1E}".format(df["trip_dep_time_rmse"][-1]))

        print("Average travel time: {:.4f}s".format(df["road_leg_travel_time_mean"][-1]))
        print("Surplus: {:.4f}".format(df["surplus_mean"][-1]))
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
            label=r"$N = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{dep}}$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_nb_agents_dep_time_rmse.pdf"))

    # Expected travel time diff. RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["road_leg_exp_travel_time_diff_rmse"],
            alpha=0.7,
            color=mpl_utils.CMP(i),
            label=r"$N = ${}".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{tt}}$ (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_nb_agents_exp_travel_time_diff_rmse.pdf"))

    # Expected travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["exp_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$N = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel("Expected travel-time function RMSE (seconds)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_nb_agents_exp_weights_rmse.pdf"))

    # Simulated travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["sim_road_network_weights_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$N = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^T$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_nb_agents_sim_weights_rmse.pdf"))
