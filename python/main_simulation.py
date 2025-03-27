# The goal of this script is to run a METROPOLIS2 simulation and compare the results to the theoretical
# solution.
import os

import numpy as np

import functions
import mpl_utils
import theoretical_solution

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/main_simulation/"

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
        functions.save_agents(RUN_DIR, nb_agents=N, departure_time_mu=MU)

        print("Writing road network")
        functions.save_road_network(RUN_DIR, bottleneck_flow=BOTTLENECK_FLOW)

        print("Writing parameters")
        functions.save_parameters(
            RUN_DIR,
            learning_value=LEARNING_VALUE,
            nb_iterations=NB_ITERATIONS,
            recording_interval=RECORDING_INTERVAL,
        )

        print("Running simulation")
        functions.run_simulation(RUN_DIR)

    print("Reading simulation results")
    print("Running time: {:.2f} s".format(functions.read_running_time(RUN_DIR)))
    iter_df = functions.read_iteration_results(RUN_DIR)
    print(
        "Exp. travel time diff RMSE: {:.1E}".format(
            iter_df["road_trip_exp_travel_time_diff_rmse"][-1]
        )
    )
    print("Exp condition RMSE: {:.1E}".format(iter_df["exp_road_network_cond_rmse"][-1]))
    if iter_df["alt_dep_time_rmse"][-1]:
        print("Dep. time RMSE: {:.1E}".format(iter_df["alt_dep_time_rmse"][-1]))

    df = functions.read_leg_results(RUN_DIR)

    ttf = functions.read_net_cond_sim_edge_ttfs(RUN_DIR)

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
    print("Average travel time: {:.4f}s".format(iter_df["road_trip_travel_time_mean"][-1]))
    print("Surplus: {:.4f}".format(iter_df["surplus_mean"][-1]))

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
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_dep_rate.pdf"))

    # Departure and arrival rate.
    bins = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 200)
    ts2 = (bins[1:] + bins[:-1]) / 2
    simulated_td_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
    simulated_ta_rs, _ = np.histogram((df["departure_time"] + df["travel_time"]).to_numpy(), bins=bins, density=True)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        ts2,
        N * simulated_td_rs,
        color=mpl_utils.CMP(1),
        alpha=0.7,
        label="Departure rate",
    )
    ax.plot(
        ts2,
        N * simulated_ta_rs,
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label="Arrival rate",
    )
    ax.legend()
    ax.set_xlabel("Departure time $t^d$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Frequency (car / second)")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + [functions.TSTAR]
    time_labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + ["$t^*$"]
    ax.set_xticks(all_times, labels=time_labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_dep_arr_rate.png"), dpi=300)

    # Travel time scatter.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.scatter(
        df["departure_time"],
        df["travel_time"],
        s=3,
        color=mpl_utils.CMP(0),
        alpha=0.9,
    )
    ax.set_xlabel("Departure time $t^d$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Travel time $t^a - t^d$")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + [functions.TSTAR]
    time_labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + ["$t^*$"]
    ax.set_xticks(all_times, labels=time_labels)
    fig.tight_layout()
    fig.savefig(
        os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_travel_time_scatter.png"), dpi=300
    )

    # Cumulative distribution.
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
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_cum_dep_rate.pdf"))

    # Travel-time function.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        ttf["departure_time"],
        ttf["travel_time"],
        "-o",
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label="Interpolated function",
    )
    ax.scatter(
        df["departure_time"],
        df["travel_time"],
        s=1,
        color=mpl_utils.CMP(2),
        alpha=0.1,
        label="Observation",
    )
    ax.axvline(functions.TSTAR, color="black", linestyle="dashed")
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Travel time $T(t)$ (seconds)")
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + [functions.TSTAR]
    time_labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + ["$t^*$"]
    ax.set_xticks(all_times, labels=time_labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_travel_time_function.png"))

    # RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    ax.plot(
        xs,
        iter_df["alt_dep_time_rmse"],
        alpha=0.7,
        color=mpl_utils.CMP(0),
        label=r"$\text{RMSE}_{\kappa}^{\text{dep}}$",
    )
    ax.plot(
        xs,
        iter_df["exp_road_network_cond_rmse"],
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
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "main_simulation_rmse.pdf"))
