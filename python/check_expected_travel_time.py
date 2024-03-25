# The goal of this script is to run a METROPOLIS2 simulation with 2 iterations and fixed departure
# times.
# Because departure times are fixed, the travel times will be the same for the 2 iterations.
# We use naive learning which means that the travel times anticipated for the second iteration are
# the travel times of the first iteration.
# Therefore, the travel times are supposed to be well anticipated for the second iteration.
import os

import functions

# Path to the directory where the simulation input should be stored.
RUN_DIR = "./runs/expected_travel_time/"

# Number of agents to simulate.
N = 100000

if __name__ == "__main__":
    if not os.path.isdir(RUN_DIR):
        os.makedirs(RUN_DIR)

    print("Writing agents")
    functions.save_agents(
        RUN_DIR,
        nb_agents=N,
        exogenous_departure_time=True,
        exogenous_departure_time_period=[7 * 3600 + 900, 8 * 3600 + 2700],
    )

    print("Writing road network")
    functions.save_road_network(RUN_DIR, bottleneck_flow=N / 2000)

    print("Writing parameters")
    functions.save_parameters(RUN_DIR, learning_value=0.0, nb_iterations=2, recording_interval=1.0)

    print("Running simulation")
    functions.run_simulation(RUN_DIR)

    print("Reading results")
    df = functions.read_leg_results(RUN_DIR)
    print(
        "Difference between simulated and expected travel time:\n{}".format(
            df["tt_diff"].describe()
        )
    )
    print(
        "Absolute difference between simulated and expected travel time:\n{}".format(
            df["tt_diff"].abs().describe()
        )
    )
