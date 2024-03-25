# The goal of this script is to run a METROPOLIS2 simulation and plot a graph of the simulated
# travel-time function.
import os

import numpy as np
import polars as pl

import functions
import mpl_utils

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/ttf/"

# Number of agents to simulate.
N = 200
# Scale of the departure-time epsilons.
MU = 1.0
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.
# Parameter of the learning model: weight of the current iteration relative to the previous value.
LEARNING_VALUE = 0.5
# Number of iterations to run.
NB_ITERATIONS = 200
# Interval of time, in seconds, between two breakpoints for the travel-time function.
RECORDING_INTERVAL = 240.0
# If `True`, analyze the results without running the simulation (only works if the simulation was
# already run before).
SKIP_RUN = False


def seconds_to_time_str(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    return f"{h}:{m:02} a.m."


if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    if not SKIP_RUN:
        print("Writing agents")
        functions.save_agents(RUN_DIR, N, departure_time_mu=MU)

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
    df = functions.read_leg_results(RUN_DIR)

    weights = functions.read_net_cond_sim_edge_ttfs(RUN_DIR)
    if weights["departure_time"][-1] != functions.PERIOD[1]:
        weights = weights.vstack(
            pl.DataFrame(
                {"departure_time": functions.PERIOD[1], "travel_time": weights["travel_time"][-1]}
            )
        )

    # Travel-time function.
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.scatter(
        df["departure_time"],
        df["travel_time"],
        s=2,
        color=mpl_utils.CMP(0),
        label=r"$(t^{\text{d}}_n, t^{\text{a}}_n - t^{\text{d}}_n)$",
    )
    ax.plot(
        weights["departure_time"],
        weights["travel_time"],
        "--o",
        markersize=4,
        color=mpl_utils.CMP(1),
        label="$T(t)$",
    )
    ax.legend()
    i = 3
    x0 = weights["departure_time"][i]
    y0 = weights["travel_time"][i]
    x1 = weights["departure_time"][i + 1]
    y1 = weights["travel_time"][i + 1]
    y_min = min(y0, y1)
    y_max = max(y0, y1)
    ax.plot([x0, x1, x1], [y_min, y_min, y_max], linestyle="dotted", color="black")
    ax.annotate(r"$\delta$", ((x0 + x1) / 2, y_min), ha="center", va="top")
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(functions.PERIOD[0], functions.PERIOD[1])
    ax.set_ylabel("Travel time (seconds)")
    ax.set_ylim(bottom=0)
    xticks = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 7)
    labels = [seconds_to_time_str(t) for t in xticks]
    ax.set_xticks(xticks, labels=labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "ttf_example.pdf"))
