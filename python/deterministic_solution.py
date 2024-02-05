# Compute and plot graph of the analytical solution of the deterministic bottleneck model (Arnott et
# al., 1990).
import os

import numpy as np

import functions
import mpl_utils

# Time window of feasible departure times (in seconds).
PERIOD = [7.0 * 3600.0, 8.0 * 3600.0]
# Number of agents.
N = 100_000
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.

if __name__ == "__main__":
    # `t_first` is the time at which congestion starts.
    t_first = (
        functions.TSTAR
        - (functions.GAMMA / (functions.BETA + functions.GAMMA)) * N / BOTTLENECK_FLOW
    )
    # `t_last` is the time at which congestion ends.
    t_last = (
        functions.TSTAR
        + (functions.BETA / (functions.BETA + functions.GAMMA)) * N / BOTTLENECK_FLOW
    )
    # `t_tilde` is the on-time departure time.
    t_tilde = (
        functions.TSTAR
        - (
            functions.BETA
            * functions.GAMMA
            / (functions.ALPHA * (functions.BETA + functions.GAMMA))
        )
        * N
        / BOTTLENECK_FLOW
    )

    # `r0` is the departure rate from `t_first` to `t_tilde`.
    r0 = BOTTLENECK_FLOW + functions.BETA * BOTTLENECK_FLOW / (functions.ALPHA - functions.BETA)
    # `r1` is the departure rate from `t_tilde` to `t_last`.
    r1 = BOTTLENECK_FLOW - functions.GAMMA * BOTTLENECK_FLOW / (functions.ALPHA + functions.GAMMA)

    # Compute equilibrium utility level for any departure time.
    ts = np.linspace(PERIOD[0], PERIOD[1], 200)
    utilities = -functions.BETA / 3600 * np.maximum(
        0, functions.TSTAR - ts
    ) - functions.GAMMA / 3600 * np.maximum(0, ts - functions.TSTAR)
    eq_utility = -functions.BETA / 3600 * (functions.TSTAR - t_first)
    utilities = np.minimum(utilities, eq_utility)

    # Create a graph of departure rate, arrival rate and equilibrium cost.
    fig, ax1 = mpl_utils.get_figure(fraction=0.8)
    #  for t in (t_first, t_tilde, t_last):
    #  ax1.axvline(t, color="black", linestyle="solid", alpha=1, linewidth=1)
    # Departure rate.
    lns1 = ax1.plot(
        [PERIOD[0], t_first, t_first, t_tilde, t_tilde, t_last, t_last, PERIOD[1]],
        [0, 0, r0, r0, r1, r1, 0, 0],
        alpha=0.7,
        color=mpl_utils.CMP(0),
        linestyle="dashed",
        label="Departure rate (left axis)",
    )
    # Arrival rate.
    lns2 = ax1.plot(
        [PERIOD[0], t_first, t_first, t_last, t_last, PERIOD[1]],
        [0, 0, BOTTLENECK_FLOW, BOTTLENECK_FLOW, 0, 0],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        linestyle="dotted",
        label="Arrival rate (left axis)",
    )
    ax2 = ax1.twinx()
    # Equilibrium utility.
    lns3 = ax2.plot(
        ts,
        utilities,
        alpha=0.7,
        color=mpl_utils.CMP(2),
        linestyle="solid",
        label="Utility (right axis)",
    )
    ax1.set_xlim(PERIOD[0], PERIOD[1])
    ax1.set_ylim(bottom=0)
    ax1.set_xlabel("Departure time")
    ax1.set_ylabel("Departure / arrival rate")
    ax2.set_ylim(-5, 0)
    ax2.set_ylabel("Utility")
    ax1.set_xticks([t_first, t_tilde, t_last], labels=[r"$t_q$", r"$\tilde{t}$", r"$t_{q'}$"])
    lns = lns1 + lns2 + lns3
    labels = [l.get_label() for l in lns]
    ax1.legend(lns, labels)
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "analytical_solution.pdf"))
