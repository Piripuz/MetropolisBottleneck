# The goal of this script is to run a METROPOLIS2 simulation with uniform epsilons and compare the
# results to the theoretical solution.
import os
import json

import numpy as np

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/uniform_epsilons/"
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
        print("Writing agents")
        agents = functions.get_agents(N, departure_time_mu=MU, uniform_epsilons=True)
        with open(os.path.join(RUN_DIR, "agents.json"), "w") as f:
            f.write(json.dumps(agents))

        print("Writing road network")
        road_network = functions.get_road_network(bottleneck_flow=BOTTLENECK_FLOW)
        with open(os.path.join(RUN_DIR, "road-network.json"), "w") as f:
            f.write(json.dumps(road_network))

        print("Writing parameters")
        parameters = functions.get_parameters(
            learning_value=LEARNING_VALUE,
            nb_iteration=NB_ITERATIONS,
            recording_interval=RECORDING_INTERVAL,
        )
        with open(os.path.join(RUN_DIR, "parameters.json"), "w") as f:
            f.write(json.dumps(parameters))

        print("Running simulation")
        functions.run_simulation(RUN_DIR)

    print("Reading simulation results")
    print("Running time: {:.2f} s".format(functions.read_running_time(RUN_DIR)))
    iter_df = functions.read_iteration_results(RUN_DIR)
    print(
        "Exp. travel time diff RMSE: {:.1E}".format(
            iter_df["road_leg_exp_travel_time_diff_rmse"][-1]
        )
    )
    print("Exp weight RMSE: {:.1E}".format(iter_df["exp_road_network_weights_rmse"][-1]))
    print("Dep. time RMSE: {:.1E}".format(iter_df["trip_dep_time_rmse"][-1]))

    df = functions.read_leg_results(RUN_DIR)

    weights = functions.read_sim_weight_results(RUN_DIR)
    simulated_tt = weights["points"]

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

    print("Analyzing results")
    # Aggregate statistics.
    print("Average travel time: {:.4f}s".format(iter_df["road_leg_travel_time_mean"][-1]))
    print("Surplus: {:.4f}".format(df["surplus_mean"][-1]))

    # Computing distance between theory and simulation.
    D = theoretical_solution.distance_theoretical(df, times, denominator, parameters)
    print("D = {:.4%}".format(D))

    print("Plotting graphs")
    # Departure rate.
    bins = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 300)
    ts = (bins[1:] + bins[:-1]) / 2
    theoretical_rs = [theoretical_solution.dep_rate(t, denominator, times, parameters) for t in ts]
    simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
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
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "uniform_epsilons_dep_rate.pdf"))

    # Cumulative departure rate.
    ys_simulated = np.cumsum(simulated_rs) * 3600 / len(ts)
    ys_theoretical = (np.cumsum(theoretical_rs) / N) * 3600 / len(ts)
    #  print(np.max(np.abs(ys_simulated - ys_theoretical)))
    t_max = np.argmax(np.abs(ys_simulated - ys_theoretical))
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for _, t in times.items():
        ax.axvline(t, color="black", linestyle="dashed")
    ax.plot(
        ts,
        ys_simulated,
        color=mpl_utils.CMP(1),
        alpha=0.7,
        label="Simulated",
    )
    ax.plot(
        ts,
        ys_theoretical,
        linestyle="dashed",
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label="Analytical",
    )
    ax.plot([ts[t_max], ts[t_max]], [ys_simulated[t_max], ys_simulated[t_max]], color="red")
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Cumulative rate of departures from origin")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + list(times.values())
    time_labels = theoretical_solution.get_time_labels(times)
    labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + time_labels
    ax.set_xticks(all_times, labels=labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "uniform_epsilons_cum_dep_rate.pdf"))

    # Travel-time function.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for _, t in times.items():
        ax.plot(
            [t, t],
            [0, theoretical_solution.travel_time(t, denominator, times, parameters)],
            color="black",
            linestyle="dashed",
        )
    ts = np.arange(
        functions.PERIOD[0],
        functions.PERIOD[0] + len(weights["points"]) * weights["interval_x"],
        weights["interval_x"],
    )
    tts = np.fromiter(
        (theoretical_solution.travel_time(t, denominator, times, parameters) for t in ts),
        dtype=np.float64,
    )
    ax.plot(
        ts,
        weights["points"],
        color=mpl_utils.CMP(1),
        alpha=0.7,
        label="Simulated",
    )
    ax.plot(
        ts,
        tts,
        linestyle="dashed",
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label="Analytical",
    )
    m = max(weights["points"]) - np.max(tts)
    print("Difference between maximum simulated and theoretical travel time: {:.2f}s".format(m))
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
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "uniform_epsilons_travel_time_function.pdf"))

    # RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    ax.plot(
        xs,
        iter_df["trip_dep_time_rmse"],
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label=r"$\text{RMSE}_{\kappa}^{\text{dep}}$",
    )
    ax.plot(
        xs,
        iter_df["exp_road_network_weights_rmse"],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        label=r"$\text{RMSE}_{\kappa}^T$",
    )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel("Convergence (seconds, log scale)")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "uniform_epsilons_rmse.pdf"))

    # Compare epsilon distributions.
    rng = np.random.default_rng(functions.RANDOM_SEED)
    random_epsilons = rng.uniform(size=N)
    uniform_epsilons = np.arange(0.0, 1.0, 1 / N)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    bins = np.linspace(0, 1, 20)
    ax.hist(
        random_epsilons, bins=bins, label="Random sampling", alpha=0.5, density=True, rwidth=0.95
    )
    ax.hist(
        uniform_epsilons,
        bins=bins,
        label="Systematic sampling",
        alpha=0.5,
        density=True,
        rwidth=0.95,
    )
    ax.set_xlabel("$u_n$")
    ax.set_ylabel("Frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "uniform_vs_random_epsilons.pdf"))
