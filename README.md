# Bottleneck model in METROPOLIS2

This repository contains Python scripts to simulate the Bottleneck model with the METROPOLIS2
transport simulator.
The model replicated is the one from de Palma, Ben-Akiva, Lefevre and Litinas (1983).
The results from these scripts are presented in Javaudin and de Palma (2024).

de Palma, A., Ben-Akiva, M., Lefevre, C., & Litinas, N. (1983). Stochastic equilibrium model of peak period traffic congestion. _Transportation Science, 17_(4), 430-453.

Javaudin, L., & de Palma, A. (2024). METROPOLIS2: A Multi-Modal Agent-Based Transport Simulator.
_THEMA Working Paper_.

## How to run

The location to the METROPOLIS2 executable is defined in the Python file `python/functions.py`
(variable METROSIM_EXEC).
Version 0.8.0 of METROPOLIS2 was used in the companion paper, Javaudin and de Palma (2024).

The graphs are stored in the `graph/` directory.
The runs input and output data are stored in the `runs/` directory.

The list of Python packages required to run the code are listed in `requirements.txt`.

The Python code is located in the `python/` directory.
Configuration variables are defined as global variables at the beginning of the scripts.

Library files:

- `functions.py`: Functions used to generate the input data for METROPOLIS2,
  many variables can be configured there.
- `mpl_utils.py`: Functions used to plot graphs with `matplotlib`.

Script files:

- `main_simulation.py`: Run the standard simulation and plot the results.
- `different_epsilons.py`: Run simulations with different random seeds.
- `uniform_epsilons.py`: Run the standard simulation with systematic sampling.
- `distributed_tstars.py`: Run a simulation where desired arrival times are distributed among
  agents.
- `convergence_learning_model.py`: Run simulations with different values for the smoothing factor,
  $\lambda$.
- `impact_breakpoint_interval.py`: Run simulations with different values for the length between two
  breakpoints, $\delta$.
- `impact_mu.py`: Run simulations with different values for the scale of utility's random
  component, $\mu$.
- `impact_nb_agents.py`: Run simulations with different values for the number of agents, $N$.
- `theoretical_solution.py`: Functions to compute the analytical solution of the model, also plot
  graphs of the solution.
- `deterministic_solution.py`: Compute and plot the analytical solution of the deterministic model.
- `bottleneck_example.py`: Plot a graph to illustrate a travel-time function in METROPOLIS2.
- `mu_interpretation.py`: Compute values and plot graph to interpret the scale of utility's random
  component, $\mu$.
