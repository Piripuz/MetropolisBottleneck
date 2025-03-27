# How METROPOLIS2 convergence varies with mu.
import os

import numpy as np

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/impact_mu/"
# Values to test for mu.
PARAMETERS = [0.2, 0.5, 1.0, 2.0, 5.0]
# Parameter of the learning model.
LEARNING_VALUE = {
    0.2: 0.01,
    0.5: 0.1,
    1.0: 0.4,
    2.0: 0.8,
    5.0: 1.0,
}
# Number of iterations to run.
NB_ITERATIONS = 1000
# Number of agents to simulate.
N = 100_000
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
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
            print(f"\n=====\nNew parameter value: {parameter}\n=====\n")
            directory = os.path.join(RUN_DIR, f"{i}")
            if not os.path.isdir(directory):
                os.makedirs(directory)

            print("Writing input")
            functions.save_agents(directory, nb_agents=N, departure_time_mu=parameter)
            functions.save_road_network(directory, bottleneck_flow=BOTTLENECK_FLOW)
            functions.save_parameters(
                directory,
                learning_value=LEARNING_VALUE[parameter],
                nb_iterations=NB_ITERATIONS,
                recording_interval=RECORDING_INTERVAL,
            )

            print("Running simulation")
            functions.run_simulation(directory)

    print("Computing theoretical results")
    parameters = {
        "n": N,
        "bottleneck_flow": BOTTLENECK_FLOW,
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
    theoretical_results = list()
    for i, parameter in enumerate(PARAMETERS):
        print("===== μ = {} =====".format(parameter))
        directory = os.path.join(RUN_DIR, f"{i}")

        print("Running time: {:.2f} s".format(functions.read_running_time(directory)))

        # Theoretical results.
        parameters["mu"] = parameter
        times, denominator = theoretical_solution.equilibrium(parameters)
        theoretical_results.append((times, denominator))

        df = functions.read_leg_results(directory)
        D = theoretical_solution.distance_theoretical(df, times, denominator, parameters)
        print("D = {:.4%}".format(D))

        df = functions.read_iteration_results(directory)
        iteration_results.append(df)
        print("Exp. travel time RMSE: {:.1E}".format(df["road_trip_exp_travel_time_diff_rmse"][-1]))
        print("Exp weight RMSE: {:.1E}".format(df["exp_road_network_cond_rmse"][-1]))
        print("Dep. time RMSE: {:.1E}".format(df["alt_dep_time_rmse"][-1]))

        print("Average travel time: {:.4f}s".format(df["road_trip_travel_time_mean"][-1]))
        print("Surplus: {:.4f}".format(df["surplus_mean"][-1]))

    # Departure-time RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["alt_dep_time_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\mu = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{dep}}$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_mu_dep_time_rmse.pdf"))

    # Expected travel time diff. RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["road_trip_exp_travel_time_diff_rmse"],
            alpha=0.7,
            color=mpl_utils.CMP(i),
            label=r"$\mu = ${}".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{tt}}$ (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_mu_exp_travel_time_diff_rmse.pdf"))

    # Expected travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["exp_road_network_cond_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\mu = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^T$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_mu_exp_weights_rmse.pdf"))

    # Simulated travel-time function RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (parameter, df) in enumerate(zip(PARAMETERS, iteration_results)):
        ax.plot(
            xs,
            df["sim_road_network_cond_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i),
            label=r"$\mu = {}$".format(parameter),
        )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel("Simulated travel-time function RMSE (seconds)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_mu_sim_weights_rmse.pdf"))

    # Theoretical departure rate.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ts = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 300)
    for i, (parameter, (times, denominator)) in enumerate(zip(PARAMETERS, theoretical_results)):
        parameters["mu"] = parameter
        r = np.fromiter(
            (theoretical_solution.dep_rate(t, denominator, times, parameters) for t in ts),
            dtype=np.float64,
        )
        ax.plot(ts, r, alpha=0.7, color=mpl_utils.CMP(i), label=r"$\mu = {}$".format(parameter))
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel(r"Rate of departures from origin $r^{\text{d}}(t)$")
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([functions.seconds_to_time_str(t) for t in ax.get_xticks()])
    ax.legend()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "impact_mu_dep_rate.pdf"))
