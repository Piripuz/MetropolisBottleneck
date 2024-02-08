# This script computes and plots the theoretical solution of the stochastic bottleneck model (de
# Palma, Ben-Akiva, Lefevre, and Litinas, 1983).
import os

import numpy as np
from scipy import optimize

import mpl_utils
from functions import seconds_to_time_str

np.seterr("raise")

# Number of agents to simulate.
N = 100_000
# Value of mu for the departure-time choice model.
MU = 1.0
# Bottleneck flow, in number of vehicles per second.
BOTTLENECK_FLOW = N / (3600.0 * 2 / 3)  # All agents can go through in 40 minutes.

KEY_TO_LABEL = {
    "t_start": "$t_q$",
    "t_tilde": r"$\tilde{t}$",
    "t_hat": r"$\hat{t}$",
    "t_end": "$t_{q'}$",
}


def dep_rate(t, denominator, times, params):
    """Returns the departure rate from origin at time `t`, given the value of the denominator `E`
    (equation 11), the time thresholds and the parameters.
    The departure rate is computed from either equation (27) or equation (45), according to the
    value of `t`.
    """
    # If `t_hat` is not in `times`, it is equal to `t_tilde`.
    t_hat = times.get("t_hat", times["t_tilde"])
    if t <= times["t_start"]:
        theta = 1.0 * (t + params["tt0"] <= params["tstar"] - params["delta"]) - (
            params["gamma"] / params["beta"]
        ) * (t + params["tt0"] >= params["tstar"] + params["delta"])
        to_be_exped = (
            params["beta"]
            * (np.abs(theta) * params["delta"] - theta * (params["tstar"] - params["tt0"]))
            / params["mu"]
        )
        if to_be_exped < -700.0:
            # Underflow.
            r = 0
        else:
            a = (params["n"] / denominator) * np.exp(to_be_exped)
            r = a * np.exp(theta * params["beta"] * t / params["mu"])
    elif t <= times["t_tilde"]:
        dep_rate_t_start = params["bottleneck_flow"]
        k = (
            1 / dep_rate_t_start
            - (params["alpha"] - params["beta"]) / (params["alpha"] * params["bottleneck_flow"])
        ) * np.exp(params["alpha"] * (times["t_start"] - t) / params["mu"])
        r = (
            params["alpha"]
            * params["bottleneck_flow"]
            / ((params["alpha"] - params["beta"]) + params["alpha"] * params["bottleneck_flow"] * k)
        )
    elif t <= t_hat:
        k = (
            1 / dep_rate(times["t_tilde"], denominator, times, params)
            - 1 / params["bottleneck_flow"]
        ) * np.exp(params["alpha"] * (times["t_tilde"] - t) / params["mu"])
        r = params["bottleneck_flow"] / (1 + params["bottleneck_flow"] * k)
    elif t <= times["t_end"]:
        k = (
            1 / dep_rate(t_hat, denominator, times, params)
            - (params["alpha"] + params["gamma"]) / (params["alpha"] * params["bottleneck_flow"])
        ) * np.exp(params["alpha"] * (t_hat - t) / params["mu"])
        r = (
            params["alpha"]
            * params["bottleneck_flow"]
            / (
                (params["alpha"] + params["gamma"])
                + params["alpha"] * params["bottleneck_flow"] * k
            )
        )
    else:
        r = (params["n"] / denominator) * np.exp(
            params["gamma"] * (params["tstar"] + params["delta"] - t - params["tt0"]) / params["mu"]
        )
    assert r >= 0.0
    return r


def queue_length(t, denominator, times, params):
    """Returns the queue length of the bottleneck at time `t`, given the value of the denominator
    `E` (equation 11), the time thresholds and the parameters.
    The queue length is computed from equation (51).
    """
    # If `t_hat` is not in `times`, it is equal to `t_tilde`.
    t_hat = times.get("t_hat", times["t_tilde"])
    if t <= times["t_start"]:
        # Congestion has not started.
        return 0.0
    elif t <= times["t_tilde"]:
        r = dep_rate(t, denominator, times, params)
        r_t_start = params["bottleneck_flow"]
        return (params["mu"] * params["bottleneck_flow"] / (params["alpha"] - params["beta"])) * (
            params["beta"] / params["mu"] * (t - times["t_start"]) - np.log(r / r_t_start)
        )
    elif t <= t_hat:
        d0 = queue_length(times["t_tilde"], denominator, times, params)
        r = dep_rate(t, denominator, times, params)
        r_t_tilde = dep_rate(times["t_tilde"], denominator, times, params)
        return d0 - (params["mu"] * params["bottleneck_flow"] / params["alpha"]) * np.log(
            r / r_t_tilde
        )
    elif t <= times["t_end"]:
        d0 = queue_length(t_hat, denominator, times, params)
        r = dep_rate(t, denominator, times, params)
        r_t_hat = dep_rate(t_hat, denominator, times, params)
        return d0 + (
            params["mu"] * params["bottleneck_flow"] / (params["alpha"] + params["gamma"])
        ) * (-params["gamma"] / params["mu"] * (t - t_hat) - np.log(r / r_t_hat))
    else:
        # Congestion is finished.
        return 0.0


def travel_time(t, denominator, times, params):
    """Returns the travel time from origin to destination at time `t`."""
    return params["tt0"] + queue_length(t, denominator, times, params) / params["bottleneck_flow"]


def equilibrium_conditional(denominator, params):
    """Returns the departure-rate and the travel-time functions given the value of `E`."""
    times = {
        "t_start": params["period"][1],
        "t_tilde": params["tstar"] - params["delta"] - params["tt0"],
        "t_hat": params["tstar"] + params["delta"] - params["tt0"],
        "t_end": params["period"][1],
    }

    if params["n"] / denominator <= params["bottleneck_flow"]:
        # Congestion level is never reached.
        return times

    # `t_start` is the time at which the congestion starts, i.e., the departure rate goes above the
    # bottleneck capacity.
    # We find `t_start` by inverting `r(t) = s`, using equation (27).
    a = (params["n"] / denominator) * np.exp(
        max(
            params["beta"] * (params["tt0"] - params["tstar"] + params["delta"]) / params["mu"],
            -700,
        )
    )
    t_start = (params["mu"] / params["beta"]) * np.log(params["bottleneck_flow"] / a)

    assert t_start <= times["t_tilde"], "Congestion cannot start after earliest departures"

    if t_start < params["period"][0]:
        # Congestion starts immediately.
        t_start = params["period"][0]

    times["t_start"] = t_start

    if t_start < times["t_tilde"]:
        # `t_tilde` is the earliest on-time departure (equation 14).
        times["t_tilde"] = optimize.bisect(
            lambda t: t
            + params["tt0"]
            + queue_length(t, denominator, times, params) / params["bottleneck_flow"]
            - params["tstar"]
            + params["delta"],
            times["t_start"] - 1.0,
            times["t_tilde"] + 1.0,
        )
    else:
        # The early departure period is finished before congestion starts.
        pass

    # `t_hat` is the latest on-time departure (equation 15).
    if params["delta"] > 0:
        times["t_hat"] = optimize.bisect(
            lambda t: t
            + params["tt0"]
            + queue_length(t, denominator, times, params) / params["bottleneck_flow"]
            - params["tstar"]
            - params["delta"],
            times["t_tilde"],
            params["period"][1],
        )
    else:
        times["t_hat"] = times["t_tilde"]

    # `t_end` is the time at which congestion ends, i.e., the time at which queue length reaches
    # zero again.
    if queue_length(params["period"][1], denominator, times, params) > 0.0:
        # Congestion does not end before the period ends.
        pass
    else:
        times["t_end"] = optimize.bisect(
            lambda t: queue_length(t, denominator, times, params),
            times["t_tilde"],
            params["period"][1],
        )

    if times["t_hat"] == times["t_tilde"]:
        # `t_hat` is not required if it is equal to `t_tilde`.
        times.pop("t_hat")
    return times


def integral(t, times, denominator, params):
    """Returns the integral of the departure rate from the start of the period to `t`."""
    # If `t_hat` is not in `times`, it is equal to `t_tilde`.
    t_hat = times.get("t_hat", times["t_tilde"])
    assert t <= params["period"][1]
    assert t >= params["period"][0]
    assert times["t_tilde"] <= t_hat
    assert times["t_start"] <= times["t_end"]
    # Case A: t_start <= t_tilde <= t_hat <= t_end.
    # Case B: t_tilde <= t_start <= t_end <= t_hat.
    # Case C: t_tilde <= t_hat <= t1 = t_start = t_end.
    if t <= min(times["t_start"], times["t_tilde"]):
        # No congestion, early departures.
        to_be_exped = (
            params["beta"] * (params["tt0"] - params["tstar"] + params["delta"]) / params["mu"]
        )
        if to_be_exped < -700.0:
            # Underflow: n is practically 0.
            return 0
        a = (params["n"] / denominator) * np.exp(to_be_exped)
        n = (
            (params["mu"] / params["beta"])
            * a
            * (
                np.exp(params["beta"] * t / params["mu"])
                - np.exp(params["beta"] * params["period"][0] / params["mu"])
            )
        )
        assert n >= 0
        return n
    elif t <= times["t_tilde"]:
        # Congestion, early departures.
        # Integral from `t0` to `t_start`.
        n1 = integral(times["t_start"], times, denominator, params)
        # Integral from `t_start` to `t`.
        r0 = dep_rate(times["t_start"], denominator, times, params)
        if r0 <= 0.0:
            # Underflow.
            n2 = 0
        else:
            n2 = (
                (params["mu"] * params["bottleneck_flow"])
                / (params["alpha"] - params["beta"])
                * (
                    params["alpha"] / params["mu"] * (t - times["t_start"])
                    - np.log(
                        dep_rate(t, denominator, times, params)
                        / dep_rate(times["t_start"], denominator, times, params)
                    )
                )
            )
        n = n1 + n2
        assert n >= 0
        return n
    elif t <= times["t_start"] and t <= t_hat:
        # t > t_tilde, t <= t_start, t <= t_hat
        # No congestion, on-time departures.
        # Integral from `t0` to `t_tilde`.
        n1 = integral(times["t_tilde"], times, denominator, params)
        # Integral from `t_tilde` to `t`.
        n2 = (params["n"] / denominator) * (t - times["t_tilde"])
        n = n1 + n2
        assert n >= 0
        return n
    elif t > times["t_start"] and t <= t_hat and t <= times["t_end"]:
        # t > t_tilde, t > t_start, t <= t_hat, t <= t_end
        # Congestion, on-time departures.
        t_prev = max(times["t_start"], times["t_tilde"])
        # Integral from `t0` to `t_prev`.
        n1 = integral(t_prev, times, denominator, params)
        # Integral from `t_prev` to `t`.
        n2 = (
            (params["mu"] * params["bottleneck_flow"])
            / params["alpha"]
            * (
                params["alpha"] / params["mu"] * (t - t_prev)
                - np.log(
                    dep_rate(t, denominator, times, params)
                    / dep_rate(t_prev, denominator, times, params)
                )
            )
        )
        n = n1 + n2
        assert n >= 0
        return n
    elif t > times["t_start"] and t <= times["t_end"]:
        # t > t_tilde, t > times["t_start"], t > t_hat, t <= t_end
        # Congestion, late departures.
        # Integral from `t0` to `t_hat`.
        n1 = integral(t_hat, times, denominator, params)
        # Integral from `t_hat` to `t`.
        n2 = (
            (params["mu"] * params["bottleneck_flow"])
            / (params["alpha"] + params["gamma"])
            * (
                params["alpha"] / params["mu"] * (t - t_hat)
                - np.log(
                    dep_rate(t, denominator, times, params)
                    / dep_rate(t_hat, denominator, times, params)
                )
            )
        )
        n = n1 + n2
        assert n >= 0
        return n
    elif t > times["t_end"] and t <= t_hat:
        # t > t_tilde, t > t_start, t > t_end, t <= t_hat.
        # No congestion, on-time departures.
        # Integral from `t0` to `t_end`.
        n1 = integral(times["t_end"], times, denominator, params)
        # Integral from `t_end` to `t`.
        n2 = (params["n"] / denominator) * (t - times["t_end"])
        n = n1 + n2
        assert n >= 0
        return n
    else:
        # No congestion, late departures.
        if t <= times["t_start"]:
            # Congestion never starts.
            assert times["t_start"] == times["t_end"]
            assert times["t_start"] >= params["period"][1]
            t_prev = t_hat
        else:
            t_prev = max(times["t_end"], t_hat)
        # Integral form `t0` to `t_prev`.
        n1 = integral(t_prev, times, denominator, params)
        # Integral from `t_prev` to `t`.
        a0 = (params["n"] / denominator) * np.exp(
            params["gamma"] * (params["tstar"] + params["delta"] - t - params["tt0"]) / params["mu"]
        )
        a1 = (params["n"] / denominator) * np.exp(
            params["gamma"]
            * (params["tstar"] + params["delta"] - t_prev - params["tt0"])
            / params["mu"]
        )
        n2 = -(params["mu"] / params["gamma"]) * (a0 - a1)
        n = n1 + n2
        assert n >= 0
        return n


def equilibrium(params):
    """Returns the departure-rate and the travel-time functions such that the integral of the
    departure rate over the period is equal to the number of individuals.
    """

    def diff_integral(d):
        times = equilibrium_conditional(d, params)
        n = integral(params["period"][1], times, d, params)
        return n - params["n"]

    min_v = -params["alpha"] * (params["tt0"] + params["n"] / params["bottleneck_flow"]) - max(
        params["beta"] * (params["tstar"] - params["delta"] - params["period"][0] - params["tt0"]),
        params["gamma"]
        * (
            params["period"][0]
            + params["tt0"]
            + params["n"] / params["bottleneck_flow"]
            - params["tstar"]
            - params["delta"]
        ),
    )
    min_denominator = np.exp(max(min_v / params["mu"], -700)) * (
        params["period"][1] - params["period"][0]
    )
    max_v = -params["alpha"] * params["tt0"]
    max_denominator = np.exp(max_v / params["mu"]) * (params["period"][1] - params["period"][0])
    denominator = optimize.bisect(
        diff_integral, max(1e-16, min_denominator - 1), max_denominator + 1
    )
    print(
        "Surplus (theoretical): {:.4f}".format(
            params["mu"] * (np.log(denominator) + np.euler_gamma)
        )
    )
    times = equilibrium_conditional(denominator, params)
    # If `t_hat` is not in `times`, it is equal to `t_tilde`.
    t_hat = times.get("t_hat", times["t_tilde"])
    print(
        "t_q: {}, t_tilde: {}, t_hat: {}, t_q': {}".format(
            seconds_to_time_str(times["t_start"]),
            seconds_to_time_str(times["t_tilde"]),
            seconds_to_time_str(t_hat),
            seconds_to_time_str(times["t_end"]),
        )
    )
    return times, denominator


def distance_theoretical(leg_df, times, denominator, parameters):
    dep_times = leg_df["departure_time"].sort().to_numpy()
    n = len(dep_times)
    D = 0.0
    for i, dt in enumerate(dep_times):
        D = max(
            D,
            abs(i - integral(dt, times, denominator, parameters)) / n,
        )
    return D


def get_time_labels(times):
    return [KEY_TO_LABEL[k] for k in times.keys()]


def plot_dep_rate(times, denominator, params, filename=None):
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    all_times = params["period"].copy()
    time_labels = [seconds_to_time_str(t) for t in params["period"]]
    for key, t in times.items():
        if t == params["period"][0] or t == params["period"][1]:
            continue
        ax.plot(
            [t, t],
            [0, dep_rate(t, denominator, times, params)],
            color="black",
            linestyle="dashed",
        )
        all_times.append(t)
        time_labels.append(KEY_TO_LABEL[key])
    ts = np.linspace(params["period"][0], params["period"][1], 300)
    r = np.fromiter((dep_rate(t, denominator, times, params) for t in ts), dtype=np.float64)
    ax.plot(ts, r, alpha=0.7, color=mpl_utils.CMP(0))
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(params["period"][0], params["period"][1])
    ax.set_ylabel(r"Rate of departures from origin $\bar{r}^{\text{d}}(t)$")
    ax.set_ylim(bottom=0)
    ax.set_xticks(all_times, labels=time_labels)
    if filename:
        fig.tight_layout()
        fig.savefig(filename)
    else:
        fig.show()


def plot_queue_length(times, denominator, params, filename=None):
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    all_times = params["period"].copy()
    time_labels = [seconds_to_time_str(t) for t in params["period"]]
    for key, t in times.items():
        if t == params["period"][0] or t == params["period"][1]:
            continue
        ax.plot(
            [t, t],
            [0, queue_length(t, denominator, times, params)],
            color="black",
            linestyle="dashed",
        )
        all_times.append(t)
        time_labels.append(KEY_TO_LABEL[key])
    ts = np.linspace(params["period"][0], params["period"][1], 300)
    d = np.fromiter((queue_length(t, denominator, times, params) for t in ts), dtype=np.float64)
    ax.plot(ts, d, alpha=0.7, color=mpl_utils.CMP(0))
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(params["period"][0], params["period"][1])
    ax.set_ylabel("Bottleneck's queue length $D^*(t)$")
    ax.set_ylim(bottom=0)
    ax.set_xticks(all_times, labels=time_labels)
    if filename:
        fig.tight_layout()
        fig.savefig(filename)
    else:
        fig.show()


def plot_travel_time_function(times, denominator, params, filename=None):
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    all_times = params["period"].copy()
    time_labels = [seconds_to_time_str(t) for t in params["period"]]
    for key, t in times.items():
        if t == params["period"][0] or t == params["period"][1]:
            continue
        ax.plot(
            [t, t],
            [0, travel_time(t, denominator, times, params)],
            color="black",
            linestyle="dashed",
        )
        all_times.append(t)
        time_labels.append(KEY_TO_LABEL[key])
    ts = np.linspace(params["period"][0], params["period"][1], 300)
    d = np.fromiter((travel_time(t, denominator, times, params) for t in ts), dtype=np.float64)
    ax.plot(ts, d, alpha=0.7, color=mpl_utils.CMP(0))
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(params["period"][0], params["period"][1])
    ax.set_ylabel(r"Travel time $\bar{T}(t)$ (seconds)")
    ax.set_ylim(bottom=0)
    ax.set_xticks(all_times, labels=time_labels)
    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()
    yticks = np.append(yticks, params["tt0"])
    yticklabels = np.append(yticklabels, "$t_f$")
    ax.set_yticks(yticks, labels=yticklabels)
    if filename:
        fig.tight_layout()
        fig.savefig(filename)
    else:
        fig.show()


def plot_inverse_sampling(times, denominator, params, filename=None, u=0.4):
    ts = np.linspace(params["period"][0], params["period"][1], 300)
    r = np.fromiter((dep_rate(t, denominator, times, params) for t in ts), dtype=np.float64)
    cum_distr = np.cumsum(r) / np.sum(r)
    fig, ax = mpl_utils.get_figure(fraction=0.8)
    idx = np.argmax(cum_distr >= u)
    ax.plot(
        [params["period"][0], ts[idx], ts[idx]],
        [u, u, 0],
        alpha=0.7,
        color=mpl_utils.CMP(1),
        linestyle="dashed",
    )
    ax.plot(ts, cum_distr, alpha=1.0, color=mpl_utils.CMP(0))
    ax.set_xlabel("Departure time $t$")
    ax.set_xlim(params["period"][0], params["period"][1])
    ax.set_ylabel(r"Cumulative probability $F^{\text{d}}_n(t)$")
    ax.set_ylim(0, 1)
    ax.set_xticks([ts[idx]], labels=[r"$(F^{\text{d}}_n)^{-1}(u_n)$"])
    ax.set_yticks([0, 1, u], labels=["0", "1", "$u_n$"])
    if filename:
        fig.tight_layout()
        fig.savefig(filename)
    else:
        fig.show()


if __name__ == "__main__":
    import functions

    PARAMETERS = {
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

    times, denominator = equilibrium(PARAMETERS)
    plot_dep_rate(
        times,
        denominator,
        PARAMETERS,
        os.path.join(mpl_utils.GRAPH_DIR, "theoretical_dep_rate.pdf"),
    )
    plot_queue_length(
        times,
        denominator,
        PARAMETERS,
        os.path.join(mpl_utils.GRAPH_DIR, "theoretical_queue_length.pdf"),
    )
    plot_travel_time_function(
        times,
        denominator,
        PARAMETERS,
        os.path.join(mpl_utils.GRAPH_DIR, "theoretical_travel_time_function.pdf"),
    )
    plot_inverse_sampling(
        times,
        denominator,
        PARAMETERS,
        os.path.join(mpl_utils.GRAPH_DIR, "theoretical_inverse_sampling.pdf"),
    )
