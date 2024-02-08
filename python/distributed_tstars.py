# The goal of this script is to run a METROPOLIS2 simulation with heterogeneous t*.
import os
import json

import numpy as np

import functions
import mpl_utils

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/distributed_tstars/"
# Path to the directory where the main simulation is stored (for comparison).
MAIN_DIR = "./runs/main_simulation/"

# Number of agents to simulate.
N = 100_000
# Scale of the departure-time epsilons.
MUS = [1.0, 0.6]
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
# Parameter of the learning model: weight of the current iteration relative to the previous value.
LEARNING_VALUE = 0.5
# Number of iterations to run.
NB_ITERATIONS = 200
# Interval of time, in seconds, between two breakpoints for the travel-time function.
RECORDING_INTERVAL = 60.0
# Standard deviation of the t* distribution.
TSTAR_STD = 7.5 * 60
# If `True`, analyze the results without running the simulation (only works if the simulation was
# already run before).
SKIP_RUN = False


if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    if not SKIP_RUN:
        for i, mu in enumerate(MUS):
            directory = os.path.join(RUN_DIR, str(i))
            if not os.path.isdir(directory):
                os.makedirs(directory)

            print("Writing agents")
            agents = functions.get_agents(N, departure_time_mu=mu, tstar_std=TSTAR_STD)
            tstars = [
                a["modes"][0]["value"]["legs"][0]["schedule_utility"]["value"]["t_star_low"]
                + functions.DELTA
                for a in agents
            ]
            with open(os.path.join(directory, "agents.json"), "w") as f:
                f.write(json.dumps(agents))

            # t* distribution.
            fig, ax = mpl_utils.get_figure(fraction=0.8)
            ax.hist(
                tstars,
                bins=60,
                density=True,
                alpha=0.7,
            )
            ax.set_xlabel("Desired arrival time $t^*$")
            ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
            ax.xaxis.set_major_formatter(lambda x, _: functions.seconds_to_time_str(x))
            ax.set_ylabel("Frequency")
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "distributed_tstars_hist.pdf"))

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

    print("Reading simulation results")
    iteration_results = list()
    for i, mu in enumerate(MUS):
        print("===== μ = {} =====".format(mu))
        directory = os.path.join(RUN_DIR, str(i))

        print("Running time: {:.2f} s".format(functions.read_running_time(directory)))
        iter_df = functions.read_iteration_results(directory)
        iteration_results.append(iter_df)
        print(
            "Exp. travel time diff RMSE: {:.1E}".format(
                iter_df["road_leg_exp_travel_time_diff_rmse"][-1]
            )
        )
        print("Exp weight RMSE: {:.1E}".format(iter_df["exp_road_network_weights_rmse"][-1]))
        print("Dep. time RMSE: {:.1E}".format(iter_df["trip_dep_time_rmse"][-1]))
        print("Average travel time: {:.4f}s".format(iter_df["road_leg_travel_time_mean"][-1]))
        print("Surplus: {:.4f}".format(iter_df["surplus_mean"][-1]))

    main_df = functions.read_iteration_results(MAIN_DIR)

    print("Plotting graphs")
    # Departure-time RMSE.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    xs = list(range(1, NB_ITERATIONS + 1))
    for i, (mu, df) in enumerate(zip(MUS, iteration_results)):
        ax.plot(
            xs,
            df["trip_dep_time_rmse"],
            alpha=0.5,
            color=mpl_utils.CMP(i + 1),
            label=r"Heterogeneous $\mu = {}$".format(mu),
        )
    ax.plot(
        xs,
        main_df["trip_dep_time_rmse"][:NB_ITERATIONS],
        alpha=0.5,
        color=mpl_utils.CMP(0),
        label=r"Homogeneous $\mu = 1.0$",
    )
    ax.set_xlabel("Iteration")
    ax.set_xlim(1, NB_ITERATIONS)
    ax.set_ylabel(r"$\text{RMSE}_{\kappa}^{\text{dep}}$ (seconds, log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "distributed_tstars_dep_time_rmse.pdf"))

    # Departure rate.
    bins = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 300)
    ts = (bins[1:] + bins[:-1]) / 2
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for i, mu in enumerate(MUS):
        directory = os.path.join(RUN_DIR, str(i))
        df = functions.read_leg_results(directory)
        simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
        ax.plot(
            ts,
            N * simulated_rs,
            color=mpl_utils.CMP(i + 1),
            alpha=0.7,
            label=r"Heterogeneous $\mu = {}$".format(mu),
        )
    df = functions.read_leg_results(MAIN_DIR)
    simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
    ax.plot(
        ts,
        N * simulated_rs,
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label=r"Homogeneous $\mu = 1.0$",
    )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.xaxis.set_major_formatter(lambda x, _: functions.seconds_to_time_str(x))
    ax.set_ylabel(r"Rate of departures from origin $r^{\text{d}}(t)$")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "distributed_tstars_dep_rate.pdf"))

    # Cumulative departure rate.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for i, mu in enumerate(MUS):
        directory = os.path.join(RUN_DIR, str(i))
        df = functions.read_leg_results(directory)
        simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
        ys_simulated = np.cumsum(simulated_rs) * 3600 / len(ts)
        ax.plot(
            ts,
            ys_simulated,
            color=mpl_utils.CMP(i + 1),
            alpha=0.7,
            label=r"Heterogeneous $\mu = {}$".format(mu),
        )
    df = functions.read_leg_results(MAIN_DIR)
    simulated_rs, _ = np.histogram(df["departure_time"].to_numpy(), bins=bins, density=True)
    ys_simulated = np.cumsum(simulated_rs) * 3600 / len(ts)
    ax.plot(
        ts,
        ys_simulated,
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label=r"Homogeneous $\mu = 1.0$",
    )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.xaxis.set_major_formatter(lambda x, _: functions.seconds_to_time_str(x))
    ax.set_ylabel("Cumulative rate of departures from origin")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "distributed_tstars_cum_dep_rate.pdf"))

    # Travel-time function.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for i, mu in enumerate(MUS):
        directory = os.path.join(RUN_DIR, str(i))
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
            color=mpl_utils.CMP(i + 1),
            alpha=0.7,
            label=r"Heterogeneous $\mu = {}$".format(mu),
        )
    weights = functions.read_sim_weight_results(MAIN_DIR)
    simulated_tt = weights["points"]
    ts = np.arange(
        functions.PERIOD[0],
        functions.PERIOD[0] + len(weights["points"]) * weights["interval_x"],
        weights["interval_x"],
    )
    ax.plot(
        ts,
        weights["points"],
        color=mpl_utils.CMP(0),
        alpha=0.7,
        label=r"Homogeneous $\mu = 1.0$",
    )
    ax.legend()
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.xaxis.set_major_formatter(lambda x, _: functions.seconds_to_time_str(x))
    ax.set_ylabel("Travel time $T(t)$ (seconds)")
    ax.set_ylim(bottom=0)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    yticks = np.append(yticks, functions.TT0)
    yticklabels = np.append(yticklabels, "$t_f$")
    ax.set_yticks(yticks, labels=yticklabels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "distributed_tstars_travel_time_function.pdf"))
