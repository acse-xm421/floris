import os
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI
import numpy as np
import pandas as pd
from scipy.stats import norm

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from floris.tools import WindRose
from scipy.interpolate import NearestNDInterpolator
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy
)
from floris.tools.optimization.layout_optimization.layout_optimization_pyoptsparse import (
    LayoutOptimizationPyOptSparse
)
from floris.tools.visualization import (
    calculate_horizontal_plane_with_turbines,
    visualize_cut_plane,
)


"""
This example shows a simple layout optimization using the python module Scipy.

A 4 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a random wind resource is supplied. The results
of the optimization show that the turbines are pushed to the outer corners of the boundary,
which makes sense in order to maximize the energy production by minimizing wake interactions.
"""
figpath = "examples/test_pic/test_exp/single/farms_"#angle_variance_wd/
csv_file_path = 'exp5-single.csv'
length = 2000.0
distance = 0

def load_floris(distance, variance_wd):
    # Load the default example floris object
    fi = FlorisInterface("examples/inputs/gch.yaml") # New CumulativeCurl model
    # fi = FlorisInterface("examples/inputs/emgauss.yaml") # New CumulativeCurl model

    # Specify the full wind farm layout: nominal wind farms
    # Set turbine locations to 6 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    
    # fixed farm
    layout_x, layout_y = new_layout(distance, variance_wd)
   
    # Turbine weights: we want to only optimize for the first 10 turbines, can we? others fixed?
    turbine_weights = np.zeros(len(layout_x), dtype=int)
    turbine_weights[:] = 1.0

    # Now reinitialize FLORIS layout
    fi.reinitialize(layout_x = layout_x, layout_y = layout_y)

    boundaries = [(distance, distance), (distance, distance+length), (distance+length, distance+length), (distance+length, distance), (distance, distance)]

    return fi, turbine_weights, boundaries


# def load_floris():
#     # Load the default example floris object
#     fi = FlorisInterface("examples/inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
#     # fi = FlorisInterface("examples/inputs/cc.yaml") # New CumulativeCurl model
#     # fi = FlorisInterface("examples/inputs/emgauss.yaml") # New CumulativeCurl model

#     # Specify the full wind farm layout: nominal wind farms
#     # Set turbine locations to 6 turbines in a rectangle
#     D = 126.0 # rotor diameter for the NREL 5MW
#     # layout_x = [0, 100, 200, ] # 3*D, 6 * D, 6 * D,
#     # layout_y = [0, 0.0, 0.0, ] # 4 * D, 0, 4 * D,
#     layout_x_fixed, layout_y_fixed = new_layout(distance, variance_wd)

#     layout_x = [0, 100, 50.0, 1000, 1100, 1050,] # 3*D, 6 * D, 6 * D,
#     layout_y = [0, 0.0, 1000.0, 0, 0.0, 1000.0,] # 4 * D, 0, 4 * D,
#     # layout_x = [0, 1000, 500.0, 1001, 2000, 1500,] # 3*D, 6 * D, 6 * D,
#     # layout_y = [0, 0.0, 1000.0, 0, 0.0, 1000.0, ] # 4 * D, 0, 4 * D,
#     # layout_x = [0, 0.0, 1000.0, ] # 3*D, 6 * D, 6 * D,
#     # layout_y = [0, 1000.0, 0, ] # 4 * D, 0, 4 * D, 
#     # layout_x = [0, 0, 3 * D, 3 * D, 6 * D, 6 * D,] # 
#     # layout_y = [0, 4 * D, 0, 4 * D, 0, 4 * D,] # 
    
#     # Turbine weights: we want to only optimize for the first 10 turbines, can we? others fixed?
#     turbine_weights = np.zeros(len(layout_x), dtype=int)
#     turbine_weights[:] = 1.0

#     # Now reinitialize FLORIS layout
#     fi.reinitialize(layout_x = layout_x, layout_y = layout_y)

#     return fi, turbine_weights

def load_windrose():
    # Load the wind rose information from an external file
    df = pd.read_csv("examples/inputs/wind_rose.csv")
    df = df[(df["ws"] < 22)].reset_index(drop=True)  # Reduce size
    df["freq_val"] = df["freq_val"] / df["freq_val"].sum() # Normalize wind rose frequencies

    # Now put the wind rose information in FLORIS format
    ws_windrose = df["ws"].unique()
    wd_windrose = df["wd"].unique()
    wd_grid, ws_grid = np.meshgrid(wd_windrose, ws_windrose, indexing="ij")

    # Use an interpolant to shape the 'freq_val' vector appropriately. You can
    # also use np.reshape(), but NearestNDInterpolator is more fool-proof.
    # insteresting
    freq_interpolant = NearestNDInterpolator(
        df[["ws", "wd"]], df["freq_val"]
    )
    freq = freq_interpolant(wd_grid, ws_grid)
    freq_windrose = freq / freq.sum()  # Normalize to sum to 1.0

    return ws_windrose, wd_windrose, freq_windrose

def new_layout(distance, variance_wd):
    if variance_wd == 0 or variance_wd == 15:
        x = np.linspace(distance,distance+length,10) # 3*D, 6 * D, 6 * D,
        y = np.repeat(distance+0.5*length, 10) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4 * D, 0, 4 * D,
    elif variance_wd == 30:
        x = np.append([distance], np.linspace(distance, distance + length, 8))
        x = np.append(x, [distance + length])

        y = np.append([distance,],np.repeat(distance+length, 8))
        y = np.append(y, [distance])
    elif variance_wd == 45:
        x = np.append([distance], np.linspace(distance, distance + length, 7))
        x = np.append(x, np.repeat((distance + length),2))

        y = np.append([distance,],np.repeat(distance+length, 7))
        y = np.append(y, [distance, distance+0.5*length])
    elif variance_wd == 60:
        x = np.append(np.repeat(distance,2), np.linspace(distance, distance + length, 6))
        x = np.append(x, np.repeat((distance + length),2))

        y = np.append([distance,distance + 0.5*length],np.repeat(distance+length, 6))
        y = np.append(y, [distance,distance+0.5*length])
    elif variance_wd == 75:
        x = np.append(np.repeat(distance,2), np.linspace(distance, distance + length, 5))
        x = np.append(x, np.repeat((distance + length),2))
        x = np.append(x, [distance+0.5*length])

        y = np.append([distance,distance + 0.5*length],np.repeat(distance+length, 5))
        y = np.append(y, [distance,distance+0.5*length])
        y = np.append(y, [distance])
    elif variance_wd == 90:
        x = np.append(np.repeat(distance,2), np.linspace(distance, distance + length, 4))
        x = np.append(x, np.repeat((distance + length),3))
        x = np.append(x, [distance+0.5*length])

        y = np.append([distance,distance + 0.5*length],np.repeat(distance+length, 4))
        y = np.append(y, [distance,distance+0.33*length, distance+0.67*length])
        y = np.append(y, [distance])

    return x,y

def plot_horizontal(fi, show_wd, save_name, exp_t, yaw_angles):
    # Create the plots of un-optimized layout
    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        yaw_angles=yaw_angles,
        wd=[show_wd],
        ws=[8.0],
        # x_bounds=[-5000, 1000.0],
        # y_bounds=[-5000.0, 1000.0],
    )
    fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
    wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list, title="Horizontal")
    name = figpath + "_" + save_name + "_t_" + str(exp_t) + ".png"
    plt.savefig(name)

class TerminalOutputToFile:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        pass


def run_farm(exp_t, mean_wd, variance_wd, distance, length, solver = "SLSQP"):

    # solver = solver
    output_log_name = "examples/log/" + solver + "_terminal_output.txt"
    sys.stdout = TerminalOutputToFile(output_log_name)
    
    # Load FLORIS: full farm including neighboring wind farms
    fi, turbine_weights, boundaries = load_floris(distance, variance_wd)
    nturbs = len(fi.layout_x)

    # # Load a dataframe containing the wind rose information
    # ws_windrose, wd_windrose, freq_windrose = load_windrose()
    # ws_windrose = ws_windrose + 0.001  # Deal with 0.0 m/s discrepancy
    wind_directions = np.arange(-180, 180, 5.0) #?
    wind_speeds = [8.0]

    if variance_wd == 0:
        variance_wd = 1e-5
    freq = norm.pdf(wind_directions, mean_wd, variance_wd)

    # Set values smaller than a threshold to 0
    threshold = 1e-5
    freq[freq < threshold] = 0

    freq = (
        (freq/np.sum(freq)).reshape( ( len(wind_directions), len(wind_speeds) ) )
    )

    # Create a FLORIS object for AEP calculations
    fi_WR = fi.copy()
    
    # fi_WR.reinitialize(wind_speeds=ws_windrose, wind_directions=wd_windrose)
    fi_WR.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)

    # The boundaries for the turbines, specified as vertices
    yaw_angles = np.array([[np.zeros(nturbs)]])
    show_wd = 0

    plot_horizontal(fi_WR, show_wd, "before_opt", exp_t, yaw_angles=yaw_angles)

    # Setup the optimization problem
    layout_opt = LayoutOptimizationScipy(fi_WR, boundaries, min_dist=200.0, solver=solver)

    # Run the optimization
    sol = layout_opt.optimize()
    
    # sol_value_x, sol_value_y = layout_opt.get_optimized_locs()

    # Get the resulting improvement in AEP
    print('... calcuating improvement in AEP')
    fi_WR.calculate_wake()
    base_aep = fi_WR.get_farm_AEP(freq=freq, turbine_weights=turbine_weights) / 1e6
    fi_WR.reinitialize(layout_x=sol[0], layout_y=sol[1])

    plot_horizontal(fi_WR, show_wd, "after_opt", exp_t, yaw_angles=yaw_angles)

    fi_WR.calculate_wake()
    turbine_powers = fi_WR.get_turbine_powers()
    farm_power = fi_WR.get_farm_power()
    opt_aep = fi_WR.get_farm_AEP(freq=freq, turbine_weights=turbine_weights) / 1e6
    # print("turbine_powers: ", turbine_powers)
    # print("farm_power: ", farm_power)
    # # print("base_aep: ", base_aep)
    # print("opt_aep: ", opt_aep)
    percent_gain = 100 * (opt_aep - base_aep) / base_aep

    # Print and plot the results
    print(f'Optimal layout: {sol}')
    print(
        f'Optimal layout improves AEP by {percent_gain:.1f}% '
        f'from {base_aep:.1f} MWh to {opt_aep:.1f} MWh'
    )

    sto_path = figpath + solver + "_plot_stochastic.png"
    layout_opt.plot_layout_opt_results(sto_path)

    return base_aep, opt_aep, percent_gain, sol


    # Reset standard output to the terminal
# sys.stdout.log_file.close()
# sys.stdout = sys.__stdout__

def calculate_farm_power(params):
    print("i, mean_wd, variance_wd, distance, length", params)
    i, mean_wd, variance_wd, distance, length = params

    # Replace this with your actual run_farm function call
    farm_power = run_farm(i, mean_wd=mean_wd, variance_wd=variance_wd, \
                          distance=distance, length=length)
    
    base_aep, opt_aep, percent_gain, sol = farm_power
    
    # Return the results as a tuple
    return (i, distance, mean_wd, variance_wd, base_aep, opt_aep, percent_gain, sol)#?[]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define your parameter ranges
    variance_wd_range = np.linspace(0, 90, 7)
    repeat_times = 1#?

    # Distribute parameter combinations across processes
    num_combinations = len(variance_wd_range) * repeat_times
    chunk_size = num_combinations // size
    if num_combinations % size != 0:
        raise ValueError("Number of combinations must be divisible by number of processes")
    remainder = num_combinations % size

    start_idx = rank * chunk_size
    # end_idx = (rank + 1) * chunk_size if rank < size - 1 else (rank + 1) * chunk_size

    local_params = []
    i = 0
    for variance_wd in variance_wd_range:
        for repeat in range(repeat_times):
        # i, mean_wd, variance_wd, distance, angle, length = params
            local_params.append((i, 0, variance_wd, distance, length))
            i += 1
    
    # Run the calculations in parallel
    local_results = []
    for params in local_params:
        local_results.append(calculate_farm_power(params))
    
    # Gather results from all processes
    all_results = comm.gather(local_results, root=0)
    # all_results = local_results
    print(all_results)

    # if rank == 0:
        # Assemble final results into a DataFrame
    import pandas as pd
    columns = ['i', 'distance', 'mean of wind direction', 'variance of wind direction', 
                'base power', 'opt power', 'percent_gain', 'optimal position']
    
    # Flatten the results list of lists
    results = [item for sublist in all_results for item in sublist]
    print(results)

    df = pd.DataFrame(results, columns=columns)
    print(df)

    # Write the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=True)  # Set index=True to include row index in the CSV

    print(f"DataFrame has been written to '{csv_file_path}'.")

if __name__ == "__main__":
    main()
