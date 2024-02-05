import os
import json

import subprocess
import numpy as np
import polars as pl
import zstandard as zstd

# Time window of feasible departure times (in seconds).
PERIOD = [7.0 * 3600.0, 8.0 * 3600.0]
# Seed of the random number generator.
RANDOM_SEED = 13081996
# Value of time, in monetary unit per hour.
ALPHA = 10.0
# Penalty for early arrivals, in monetary unit per hour.
BETA = 5.0
# Penalty for late arrivals, in monetary unit per hour.
GAMMA = 7.0
# Desired arrival time at destination, in seconds.
TSTAR = 7.5 * 3600.0
# Half-length of the desired arrival time window, in seconds.
DELTA = 0.0 * 60.0
# Free-flow travel time from origin to destination, in seconds.
TT0 = 30.0
# Path to the METROPOLIS2 executable.
METROPOLIS_EXEC = "./execs/metropolis"


def seconds_to_time_str(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(round(t % 60))
    return f"{h:02}:{m:02}:{s:02}"


def get_agents(
    nb_agents=1000,
    departure_time_mu=1.0,
    exogenous_departure_time=False,
    exogenous_departure_time_period=PERIOD,
    uniform_epsilons=False,
    random_seed=None,
    tstar_std=0.0,
):
    agents = list()
    if random_seed is None:
        random_seed = RANDOM_SEED
    rng = np.random.default_rng(RANDOM_SEED)
    random_dt = iter(
        rng.uniform(
            exogenous_departure_time_period[0], exogenous_departure_time_period[1], size=nb_agents
        )
    )
    if uniform_epsilons:
        random_u = iter(np.arange(0.0, 1.0, 1 / nb_agents))
    else:
        random_u = iter(rng.uniform(size=nb_agents))
    if tstar_std == 0:
        # Same t*.
        tstars = np.repeat(TSTAR, nb_agents)
    else:
        tstars = rng.normal(TSTAR, tstar_std, size=nb_agents)
    tstars_iter = iter(tstars)
    for i in range(nb_agents):
        tstar = next(tstars_iter)
        leg = {
            "class": {
                "type": "Road",
                "value": {
                    "origin": 0,
                    "destination": 1,
                    "vehicle": 0,
                },
            },
            "travel_utility": {
                "type": "Polynomial",
                "value": {
                    "b": -ALPHA / 3600.0,
                },
            },
            "schedule_utility": {
                "type": "AlphaBetaGamma",
                "value": {
                    "beta": BETA / 3600.0,
                    "gamma": GAMMA / 3600.0,
                    "t_star_low": tstar - DELTA,
                    "t_star_high": tstar + DELTA,
                },
            },
        }
        if exogenous_departure_time:
            departure_time_model = {
                "type": "Constant",
                "value": next(random_dt),
            }
        else:
            departure_time_model = {
                "type": "ContinuousChoice",
                "value": {
                    "period": PERIOD,
                    "choice_model": {
                        "type": "Logit",
                        "value": {
                            "u": next(random_u),
                            "mu": departure_time_mu,
                        },
                    },
                },
            }
        car_mode = {
            "type": "Trip",
            "value": {
                "legs": [leg],
                "departure_time_model": departure_time_model,
            },
        }
        agent = {
            "id": i,
            "modes": [car_mode],
        }
        agents.append(agent)
    if tstar_std > 0.0:
        print(
            "t* is ranged between {} and {}".format(
                seconds_to_time_str(np.min(tstars)), seconds_to_time_str(np.max(tstars))
            )
        )
    return agents


def get_parameters(learning_value=0.01, recording_interval=1.0, nb_iteration=100):
    parameters = {
        "period": PERIOD,
        "network": {
            "road_network": {
                "recording_interval": recording_interval,
                "spillback": False,
                "max_pending_duration": 0.0,
            }
        },
        "stopping_criteria": [
            {
                "type": "MaxIteration",
                "value": nb_iteration,
            }
        ],
        "saving_format": "Parquet",
    }
    if (
        learning_value == 0.0
        or isinstance(learning_value, str)
        and learning_value.lower() == "linear"
    ):
        parameters["learning_model"] = {"type": "Linear"}
    else:
        parameters["learning_model"] = {
            "type": "Exponential",
            "value": learning_value,
        }
    return parameters


def get_road_network(bottleneck_flow):
    return {
        "graph": {
            "edges": [
                [
                    0,
                    1,
                    {"id": 1, "base_speed": 1.0, "length": TT0, "bottleneck_flow": bottleneck_flow},
                ]
            ]
        },
        "vehicles": [{"length": 1.0, "pce": 1.0}],
    }


def run_simulation(directory, weights=None):
    command = [
        METROPOLIS_EXEC,
        "--agents",
        os.path.join(directory, "agents.json"),
        "--road-network",
        os.path.join(directory, "road-network.json"),
        "--parameters",
        os.path.join(directory, "parameters.json"),
        "--output",
        os.path.join(directory, "output"),
    ]
    if weights:
        command.extend(["--weights", weights])
    subprocess.run(" ".join(command), shell=True)


def read_iteration_results(directory):
    return pl.read_parquet(os.path.join(directory, "output", "iteration_results.parquet"))


def read_agent_results(directory):
    return pl.read_parquet(os.path.join(directory, "output", "agent_results.parquet"))


def read_leg_results(directory):
    df = pl.read_parquet(os.path.join(directory, "output", "leg_results.parquet"))
    df = df.with_columns(
        (pl.col("arrival_time") - pl.col("departure_time")).alias("travel_time"),
        (pl.col("exp_arrival_time") - pl.col("departure_time")).alias("exp_travel_time"),
    )
    df = df.with_columns((pl.col("travel_time") - pl.col("exp_travel_time")).alias("tt_diff"))
    return df


def read_sim_weight_results(directory):
    dctx = zstd.ZstdDecompressor()
    with open(os.path.join(directory, "output", "sim_weight_results.json.zst"), "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data["road_network"][0]["1"]


def read_exp_weight_results(directory):
    dctx = zstd.ZstdDecompressor()
    with open(os.path.join(directory, "output", "next_exp_weight_results.json.zst"), "br") as f:
        reader = dctx.stream_reader(f)
        data = json.load(reader)
    return data["road_network"][0]["1"]


def read_running_time(directory):
    with open(os.path.join(directory, "output", "running_times.json")) as f:
        data = json.load(f)
    return float(data["total"])
