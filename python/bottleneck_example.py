import os

import numpy as np

import mpl_utils

# Departure times of the agents, sorted.
TDS = sorted([1, 4, 5, 7, 11])
# Bottleneck's capacity in number of agent per time unit.
CAPA = 0.5
# Bounds of the travel-time function.
BOUNDS = [0, 15]

if __name__ == "__main__":
    # Compute arrival times and bottleneck's re-opening times.
    tas = list()
    openings = list()
    last_opening = -np.inf
    for i, td in enumerate(TDS):
        ta = int(max(td, last_opening))
        tas.append(ta)
        opening = int(ta + 1 / CAPA)
        openings.append(opening)
        last_opening = opening
        print("Agent {}: td: {}, ta: {}, opening: {}".format(i + 1, td, ta, opening))

    # Create a graph of the travel-time function.
    tts = [ta - td for td, ta in zip(TDS, tas)]
    dep_times = np.linspace(BOUNDS[0], BOUNDS[1], 300)
    prev_arr_time_ids = np.searchsorted(TDS, dep_times)
    tas_to_choose = np.array([-np.inf] + tas)
    prev_arr_times = tas_to_choose[prev_arr_time_ids]
    arr_times = np.maximum(dep_times, prev_arr_times + 1 / CAPA)
    trav_times = arr_times - dep_times
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    ax.plot(
        dep_times,
        trav_times,
        "--",
        color="black",
        alpha=0.7,
        label="travel-time function",
    )
    ax.plot(
        TDS,
        tts,
        "o",
        color="black",
        markersize=4,
        label=r"$(t^{\text{d}}_{n_i}, t^{\text{a}}_{n_i} - t^{\text{d}}_{n_i})$",
    )
    ax.set_xlim(dep_times[0], dep_times[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Departure time")
    ax.set_ylabel("Travel time")
    ax.grid()
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "bottleneck_example.pdf"))
