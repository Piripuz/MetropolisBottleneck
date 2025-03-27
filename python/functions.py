import os
import json

import subprocess
import numpy as np
import polars as pl

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
METROPOLIS_EXEC = "./execs/metropolis_cli"
# If True, the input and output files use the Parquet format, otherwise they use the CSV format.
USE_PARQUET = True

# Width of the departure-time choice intervals (for Multinomial logit departure time only).
DT_WIDTH = 60.0
# Bins of the departure-time choice intervals (for Multinomial logit departure time only).
DT_BINS = list(np.arange(PERIOD[0] + DT_WIDTH / 2, PERIOD[1], DT_WIDTH))


def seconds_to_time_str(t):
    h = int(t // 3600)
    m = int(t % 3600 // 60)
    s = int(round(t % 60))
    return f"{h:02}:{m:02}:{s:02}"


def save_agents(
    directory,
    nb_agents=1000,
    departure_time_mu=1.0,
    exogenous_departure_time=False,
    exogenous_departure_time_period=PERIOD,
    uniform_epsilons=False,
    random_seed=None,
    tstar_std=0.0,
    multinomial=False,
):
    if random_seed is None:
        random_seed = RANDOM_SEED
    rng = np.random.default_rng(random_seed)
    if uniform_epsilons:
        us = np.arange(0.0, 1.0, 1 / nb_agents)
    else:
        us = rng.uniform(size=nb_agents)
    # === Agent-level DataFrame ===
    agents = pl.DataFrame({"agent_id": pl.int_range(1, nb_agents + 1, eager=True)})
    # === Alternative-level DataFrame ===
    alternatives = pl.DataFrame(
        {
            "agent_id": pl.int_range(1, nb_agents + 1, eager=True),
            "alt_id": pl.int_range(1, nb_agents + 1, eager=True),
        }
    )
    if exogenous_departure_time:
        departure_times = rng.uniform(
            exogenous_departure_time_period[0], exogenous_departure_time_period[1], size=nb_agents
        )
        alternatives = alternatives.with_columns(
            pl.lit("Constant").alias("dt_choice.type"),
            pl.lit(departure_times).alias("dt_choice.departure_time"),
        )
    elif multinomial:
        offsets = rng.uniform(-DT_WIDTH / 2, DT_WIDTH / 2, size=nb_agents)
        alternatives = alternatives.with_columns(
            pl.lit("Discrete").alias("dt_choice.type"),
            pl.lit(DT_WIDTH).alias("dt_choice.interval"),
            pl.lit(offsets).alias("dt_choice.offset"),
            pl.lit("Logit").alias("dt_choice.model.type"),
            pl.lit(us).alias("dt_choice.model.u"),
            pl.lit(departure_time_mu).alias("dt_choice.model.mu"),
        )
    else:
        alternatives = alternatives.with_columns(
            pl.lit("Continuous").alias("dt_choice.type"),
            pl.lit("Logit").alias("dt_choice.model.type"),
            pl.lit(us).alias("dt_choice.model.u"),
            pl.lit(departure_time_mu).alias("dt_choice.model.mu"),
        )
    if tstar_std == 0:
        # Same t*.
        tstars = np.repeat(TSTAR, nb_agents)
    else:
        tstars = rng.normal(TSTAR, tstar_std, size=nb_agents)
        print(
            "t* is ranged between {} and {}".format(
                seconds_to_time_str(np.min(tstars)), seconds_to_time_str(np.max(tstars))
            )
        )
    alternatives = alternatives.with_columns(
        pl.lit(-ALPHA / 3600.0).alias("total_travel_utility.one"),
        pl.lit("AlphaBetaGamma").alias("destination_utility.type"),
        pl.lit(tstars).alias("destination_utility.tstar"),
        pl.lit(BETA / 3600.0).alias("destination_utility.beta"),
        pl.lit(GAMMA / 3600.0).alias("destination_utility.gamma"),
        pl.lit(DELTA).alias("destination_utility.delta"),
    )
    # === Trip-level DataFrame ===
    trips = pl.DataFrame(
        {
            "agent_id": pl.int_range(1, nb_agents + 1, eager=True),
            "alt_id": pl.int_range(1, nb_agents + 1, eager=True),
            "trip_id": pl.int_range(1, nb_agents + 1, eager=True),
            "class.type": "Road",
            "class.origin": 0,
            "class.destination": 1,
            "class.vehicle": 0,
        }
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if USE_PARQUET:
        agents.write_parquet(os.path.join(directory, "agents.parquet"))
        alternatives.write_parquet(os.path.join(directory, "alts.parquet"))
        trips.write_parquet(os.path.join(directory, "trips.parquet"))
    else:
        agents.write_csv(os.path.join(directory, "agents.csv"))
        alternatives.write_csv(os.path.join(directory, "alts.csv"))
        trips.write_csv(os.path.join(directory, "trips.csv"))


def save_road_network(directory, bottleneck_flow):
    edges = pl.DataFrame(
        {
            "edge_id": [1],
            "source": [0],
            "target": [1],
            "speed": [1.0],
            "length": [TT0],
            "bottleneck_flow": bottleneck_flow,
        }
    )
    vehicles = pl.DataFrame(
        {
            "vehicle_id": [0],
            "headway": [1.0],
            "pce": [1.0],
        }
    )
    if not os.path.isdir(directory):
        os.makedirs(directory)
    if USE_PARQUET:
        edges.write_parquet(os.path.join(directory, "edges.parquet"))
        vehicles.write_parquet(os.path.join(directory, "vehicles.parquet"))
    else:
        edges.write_csv(os.path.join(directory, "edges.csv"))
        vehicles.write_csv(os.path.join(directory, "vehicles.csv"))


def save_parameters(directory, learning_value=0.01, recording_interval=1.0, nb_iterations=100):
    if USE_PARQUET:
        ext = "parquet"
        fmt = "Parquet"
    else:
        ext = "csv"
        fmt = "CSV"
    parameters = {
        "input_files": {
            "agents": f"agents.{ext}",
            "alternatives": f"alts.{ext}",
            "trips": f"trips.{ext}",
            "edges": f"edges.{ext}",
            "vehicle_types": f"vehicles.{ext}",
        },
        "output_directory": "output",
        "period": PERIOD,
        "road_network": {
            "recording_interval": recording_interval,
            "spillback": False,
            "max_pending_duration": 0.0,
        },
        "max_iterations": nb_iterations,
        "saving_format": fmt,
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
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, "parameters.json"), "w") as f:
        f.write(json.dumps(parameters))


def run_simulation(directory):
    command = [
        METROPOLIS_EXEC,
        os.path.join(directory, "parameters.json"),
    ]
    subprocess.run(" ".join(command), shell=True)


def read_iteration_results(directory):
    if USE_PARQUET:
        return pl.read_parquet(os.path.join(directory, "output", "iteration_results.parquet"))
    else:
        return pl.read_csv(os.path.join(directory, "output", "iteration_results.csv"))


def read_agent_results(directory):
    if USE_PARQUET:
        return pl.read_parquet(os.path.join(directory, "output", "agent_results.parquet"))
    else:
        return pl.read_csv(os.path.join(directory, "output", "agent_results.csv"))


def read_leg_results(directory):
    if USE_PARQUET:
        df_real = pl.read_parquet(os.path.join(directory, "output", "net_cond_sim_edge_ttfs.parquet"))
        df_exp = pl.read_parquet(os.path.join(directory, "output", "net_cond_exp_edge_ttfs.parquet"))
    else:
        df_real = pl.read_csv(os.path.join(directory, "output", "net_cond_sim_edge_ttfs.parquet"))
        df_exp = pl.read_csv(os.path.join(directory, "output", "net_cond_exp_edge_ttfs.parquet"))
    df = df_real.with_columns(
        (df_exp["travel_time"] - df_real["travel_time"]).alias("exp_travel_time"),
    )
    df = df.with_columns((pl.col("travel_time") - pl.col("exp_travel_time")).alias("tt_diff"))
    return df


def read_net_cond_sim_edge_ttfs(directory):
    if USE_PARQUET:
        df = pl.read_parquet(os.path.join(directory, "output", "net_cond_sim_edge_ttfs.parquet"))
    else:
        df = pl.read_csv(os.path.join(directory, "output", "net_cond_sim_edge_ttfs.csv"))
    return df.select("departure_time", "travel_time")


def read_net_cond_exp_edge_ttfs(directory):
    if USE_PARQUET:
        df = pl.read_parquet(os.path.join(directory, "output", "net_cond_next_exp_edge_ttfs.parquet"))
    else:
        df = pl.read_csv(os.path.join(directory, "output", "net_cond_next_exp_edge_ttfs.csv"))
    return df


def read_running_time(directory):
    with open(os.path.join(directory, "output", "running_times.json")) as f:
        data = json.load(f)
    return float(data["total"]["secs"])
