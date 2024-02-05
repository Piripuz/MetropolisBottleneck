# This script computes a few values to help interpret the value of mu.
import os

import numpy as np

import mpl_utils
import functions

np.seterr("raise")

# Values to test for mu.
PARAMETERS = [0.2, 0.5, 1.0, 2.0, 5.0]
# Computes the probability to start in the given period.
START_PERIOD = [7 * 3600 + 20 * 60, 7 * 3600 + 40 * 60]


def integral_early(mu, t):
    """Returns the integral of `exp(V(t) / mu)` between `t0` and `t` (excluding value of travel
    time)."""
    assert t <= functions.TSTAR - functions.DELTA - functions.TT0
    beta = functions.BETA / 3600
    return (
        mu
        / beta
        * np.exp(-beta / mu * (functions.TSTAR - functions.DELTA - functions.TT0))
        * (np.exp(beta / mu * t) - np.exp(beta / mu * functions.PERIOD[0]))
    )


def integral_late(mu, t):
    """Returns the integral of `exp(V(t) / mu)` between `t` and `t1` (excluding value of travel
    time)."""
    assert t >= functions.TSTAR + functions.DELTA - functions.TT0
    gamma = functions.GAMMA / 3600
    return (
        -mu
        / gamma
        * np.exp(gamma / mu * (functions.TSTAR + functions.DELTA - functions.TT0))
        * (np.exp(-gamma / mu * functions.PERIOD[1]) - np.exp(-gamma / mu * t))
    )


def denominator(mu):
    """Returns the integral of `exp(V(t) / mu)` between `t0` and `t1` (excluding value fo travel
    time)."""
    early_denominator = integral_early(mu, functions.TSTAR - functions.DELTA - functions.TT0)
    late_denominator = integral_late(mu, functions.TSTAR + functions.DELTA - functions.TT0)
    return early_denominator + late_denominator + 2 * functions.DELTA


def probability_center(mu, t_first, t_last):
    """Returns the probability that the chosen departure time is between `t_first` and `t_last`
    (assuming no congestion).
    """
    early_denominator = integral_early(mu, functions.TSTAR - functions.DELTA - functions.TT0)
    late_denominator = integral_late(mu, functions.TSTAR + functions.DELTA - functions.TT0)
    deno = early_denominator + late_denominator + 2 * functions.DELTA
    if t_first < functions.TSTAR - functions.DELTA - functions.TT0:
        # `t_first` is such that arrival is early.
        first_numerator = integral_early(mu, t_first)
    elif t_first < functions.TSTAR + functions.DELTA - functions.TT0:
        # `t_first` is such that arrival is on time.
        first_numerator = (
            early_denominator + t_first - functions.TSTAR + functions.DELTA + functions.TT0
        )
    else:
        # `t_first` is such that arrival is late.
        first_numerator = (
            early_denominator
            + 2 * functions.DELTA
            + (late_denominator - integral_late(mu, t_first))
        )
    if t_last < functions.TSTAR - functions.DELTA - functions.TT0:
        # `t_last` is such that arrival is early.
        last_numerator = integral_early(mu, t_last)
    elif t_last < functions.TSTAR + functions.DELTA - functions.TT0:
        # `t_last` is such that arrival is on time.
        last_numerator = (
            early_denominator + t_last - functions.TSTAR + functions.DELTA + functions.TT0
        )
    else:
        # `t_last` is such that arrival is late.
        last_numerator = (
            early_denominator + 2 * functions.DELTA + (late_denominator - integral_late(mu, t_last))
        )
    return last_numerator / deno - first_numerator / deno


if __name__ == "__main__":
    ts = np.linspace(functions.PERIOD[0], functions.PERIOD[1], 200)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    for i, mu in enumerate(PARAMETERS):
        print("===== μ = {} =====".format(mu))
        p = probability_center(mu, START_PERIOD[0], START_PERIOD[1])
        print(
            "Probability to leave between {} and {}: {:.4%}".format(
                functions.seconds_to_time_str(START_PERIOD[0]),
                functions.seconds_to_time_str(START_PERIOD[1]),
                p,
            )
        )
        deno = denominator(mu)
        utilities = -functions.BETA / 3600 * np.maximum(
            0, functions.TSTAR - functions.DELTA - functions.TT0 - ts
        ) - functions.GAMMA / 3600 * np.maximum(
            0, ts - functions.TSTAR - functions.DELTA + functions.TT0
        )
        probs = 60 * np.exp(utilities / mu) / deno
        ax.plot(ts, probs, color=mpl_utils.CMP(i), label=r"$\mu = {}$".format(mu))
    ax.set_xlabel("Departure time")
    ax.set_ylabel("Probability")
    ax.set_xlim(*functions.PERIOD)
    ax.set_ylim(bottom=0)
    all_times = functions.PERIOD + [functions.TSTAR]
    time_labels = [functions.seconds_to_time_str(t) for t in functions.PERIOD] + ["$t^*$"]
    ax.set_xticks(all_times, labels=time_labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(mpl_utils.GRAPH_DIR, "mu_interpretation_probs.pdf"))
