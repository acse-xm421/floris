# Xuefei Mi acse-xm421
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from scipy.interpolate import NearestNDInterpolator
from floris.tools.optimization.layout_optimization.layout_optimization_farms_scipy import (
    LayoutOptimizationFarmsScipy
)
from floris.tools.visualization import (
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
figpath = "examples/test_pic/farms_"#angle_variance_wd/
csv_file_path = 'exp_test1.csv'
length = 2000.0

def load_fixed_farm():
    # Load the default example floris object
    fi_fixed = FlorisInterface("examples/inputs/gch.yaml") # New CumulativeCurl model

    # Specify the full wind farm layout: nominal wind farms
    # Set turbine locations to 6 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    
    # fixed farm
    layout_x_fixed = np.linspace(distance,distance+length,10) # 3*D, 6 * D, 6 * D,
    layout_y_fixed = np.repeat(distance+0.5*length, 10) #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # 4 * D, 0, 4 * D,
    
    # Turbine weights: we want to only optimize for the first 10 turbines, can we? others fixed?
    turbine_weights_fixed = np.zeros(len(layout_x_fixed), dtype=int)
    turbine_weights_fixed[:] = 0.0

    # Now reinitialize FLORIS layout
    fi_fixed.reinitialize(layout_x = layout_x_fixed, layout_y = layout_y_fixed)

    boundaries_fixed = [(0.0, 0.0), (0.0, length), (length, length), (length, 0.0), (0.0, 0.0)]

    return fi_fixed, turbine_weights_fixed, boundaries_fixed

def load_flexible_farm():
    # Load the default example floris object
    fi_flexible = FlorisInterface("examples/inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2

    # Specify the full wind farm layout: nominal wind farms
    # Set turbine locations to 6 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW

    # flexible farm
    layout_x_flexible = np.linspace(distance,distance+length,10) # 3*D, 6 * D, 6 * D,
    layout_y_flexible = np.random.randint(distance, distance+length+1, size=10) # 4 * D, 0, 4 * D,
    
    # Turbine weights: we want to only optimize for the first 10 turbines, can we? others fixed?
    turbine_weights_flexible = np.zeros(len(layout_x_flexible), dtype=int)
    turbine_weights_flexible[:] = 1.0

    # Now reinitialize FLORIS layout
    fi_flexible.reinitialize(layout_x = layout_x_flexible, layout_y = layout_y_flexible)

    boundaries_flexible = [(0.0, 0.0), (0.0, length), (length, length), (length, 0.0), (0.0, 0.0)]

    return fi_flexible, turbine_weights_flexible, boundaries_flexible

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

def plot_horizontal(fi, show_wd, solver, save_name, exp_t, yaw_angles):
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
    name = figpath + solver + "_" + save_name + "_show_wd_" + str(show_wd) +"t_" + str(exp_t) + ".png"
    plt.savefig(name)

# not finished yet, use when + angle
def check_condition(angle, dist, length):
    flag = False
    if (0 <= angle < 45) or (135 <= angle < 225) or (315 <= angle < 360):
        cos_value = np.cos(np.radians(angle))
        if dist * cos_value < length:
            flag = True
    else:
        sin_value = np.sin(np.radians(angle))
        if dist * sin_value < length:
            flag = True
    return flag

class TerminalOutputToFile:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        pass

def run_farm(exp_t, mean_wd, variance_wd, distance, length, angle=90, mean_ws=8.0, variance_ws=0.0, solver = "SLSQP"):
    
    fi_fixed, turbine_weights_fixed, boundaries_fixed = load_fixed_farm()
    fi_flexible, turbine_weights_flexible, boundaries_flexible = load_flexible_farm()

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
    
    nturbs = len(fi_fixed.layout_x) + len(fi_flexible.layout_x)
    
    fi_fixed.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)
    fi_flexible.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)

    # The boundaries for the turbines, specified as vertices
    length = length
    boundaries = [(distance, distance), (distance, distance+length), (distance+length, distance+length), (distance+length, distance), (distance, distance)]

    yaw_angles = np.array([[np.zeros(nturbs)]])
    show_wd = mean_wd

    nfarms = 2

    # fixed first
    fi_list = [fi_fixed, fi_flexible] # independent
    nturbs_list = [fi_fixed.floris.farm.n_turbines, fi_flexible.floris.farm.n_turbines]
    angle_list = [0, angle]
    dist_list = [0, distance]# if [0,0], overlap, if [0,>1000?], no overlap

    overlap_flag = check_condition(angle, distance, length)

    turbine_weights_fixed_ones = np.ones(nturbs_list[0], dtype=int)
    turbine_weights_fixed_zeros = np.zeros(nturbs_list[0], dtype=int)
    turbine_weights_flexible_ones = np.ones(nturbs_list[1], dtype=int)
    turbine_weights_flexible_zeros = np.zeros(nturbs_list[1], dtype=int)

    # fixed, flexible
    fixed_farm_weights = np.concatenate((turbine_weights_fixed_ones, turbine_weights_flexible_zeros))
    flexible_farm_weights = np.concatenate((turbine_weights_fixed_zeros, turbine_weights_flexible_ones))
    both_farms_weights = np.concatenate((turbine_weights_fixed_ones, turbine_weights_flexible_ones))

    # Setup the optimization problem
    wf_layout_opt = LayoutOptimizationFarmsScipy(nfarms=nfarms, fi_list = fi_list, nturbs_list=nturbs_list,\
                            angle_list=angle_list, dist_list=dist_list, boundary_1=boundaries, \
                            wind_directions=wind_directions, wind_speeds=wind_speeds, solver=solver)

    # Plot the starting wake situation
    if exp_t == 1:
        plot_horizontal(wf_layout_opt.wf, show_wd, solver, "before_opt", exp_t, yaw_angles=yaw_angles)
        
    wf_layout_opt.wf.calculate_wake(yaw_angles=yaw_angles)
    flexible_base_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=flexible_farm_weights) / 1e6
    fixed_base_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=fixed_farm_weights) / 1e6
    total_base_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=both_farms_weights) / 1e6

    print("before optimization x", wf_layout_opt.wf.layout_x)
    print("before optimization y", wf_layout_opt.wf.layout_y)
    print('... calcuating improvement in AEP')
    print("flexible_base_aep", flexible_base_aep)
    print("fixed_base_aep", fixed_base_aep)
    print("both_base_aep", total_base_aep)
    
    # Run the optimization
    sol = wf_layout_opt.optimize()

    # Get the resulting improvement in AEP
    wf_layout_opt.wf.reinitialize(layout_x=sol[0], layout_y=sol[1])

    # Plot the ending wake situation
    plot_horizontal(wf_layout_opt.wf, show_wd, solver, "after_opt", exp_t, yaw_angles=yaw_angles)
    wf_layout_opt.wf.calculate_wake(yaw_angles=yaw_angles)
    flexible_opt_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=flexible_farm_weights) / 1e6
    fixed_opt_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=fixed_farm_weights) / 1e6
    total_opt_aep = wf_layout_opt.wf.get_farm_AEP(freq=freq, turbine_weights=both_farms_weights) / 1e6

    flexible_percent_gain = 100 * (flexible_opt_aep - flexible_base_aep) / flexible_base_aep
    fixed_percent_gain = 100 * (fixed_opt_aep - fixed_base_aep) / fixed_base_aep
    total_percent_gain = 100 * (total_opt_aep - total_base_aep) / total_base_aep

    # Print and plot the results
    print(f'Optimal layout: {sol}')
    print(
        f'Optimal layout improves AEP by {total_percent_gain:.1f}% '
        f'from {total_base_aep:.1f} MWh to {total_opt_aep:.1f} MWh'
    )

    sto_path = figpath + solver + "_plot_stochastic_" + str(exp_t) + ".png"
    wf_layout_opt.plot_layout_opt_results(sto_path)

    return flexible_base_aep, flexible_opt_aep, flexible_percent_gain, fixed_base_aep, fixed_opt_aep, fixed_percent_gain, total_base_aep, total_opt_aep, total_percent_gain, overlap_flag, sol

if __name__ == "__main__":

    # Set the output directory and output logging
    output_log_name = "examples/log/terminal_output_exp_test1.txt"
    sys.stdout = TerminalOutputToFile(output_log_name) 

    # Create an empty DataFrame with specified column names
    columns = ['distance', 'angle', 'variance of wind direction', 'flexible farm base power', 'flexible farm opt power', \
               'flexible_percent_gain', 'fixed farm base power', 'fixed farm opt power', 'fixed_percent_gain', \
               'total farm base power', 'total farm opt power', 'total_percent_gain', 'overlap_flag', \
                'optimal position']
    df = pd.DataFrame(columns=columns)

    # Define the range of parameters
    angle_range = [225,] # np.arange(0, 360, 30)
    distance_range = [2100,]
    # distance_range = np.arange(length, length*2, 200)
    variance_wd_range = [60,] # np.linspace(0, 90, 7)#4
    # mean_wd_range = [90,]#np.arange(0, 360, 30)
    i = 1

    # Run the simulation for each parameter  
    for variance_wd in variance_wd_range:
        for angle in angle_range:
            for distance in distance_range:
                print("=====================================================")
                
                farm_power = run_farm(i, mean_wd=0, variance_wd=variance_wd, \
                                      distance=distance, angle=angle, length=length)
                
                flexible_base_aep, flexible_opt_aep, flexible_percent_gain, \
                    fixed_base_aep, fixed_opt_aep, fixed_percent_gain, \
                        total_base_aep, total_opt_aep, total_percent_gain, \
                            overlap_flag, sol = farm_power
                
                # Fill in the values one by one
                df.loc[i, 'distance'] = distance
                df.loc[i, 'angle'] = angle
                df.loc[i, 'variance of wind direction'] = variance_wd

                df.loc[i, 'flexible farm base power'] = flexible_base_aep
                df.loc[i, 'flexible farm opt power'] = flexible_opt_aep
                df.loc[i, 'flexible_percent_gain'] = flexible_percent_gain
                df.loc[i, 'fixed farm base power'] = fixed_base_aep
                df.loc[i, 'fixed farm opt power'] = fixed_opt_aep
                df.loc[i, 'fixed_percent_gain'] = fixed_percent_gain
                df.loc[i, 'total farm base power'] = total_base_aep
                df.loc[i, 'total farm opt power'] = total_opt_aep
                df.loc[i, 'total_percent_gain'] = total_percent_gain
                df.loc[i, "overlap_flag"] = overlap_flag
                df.loc[i, "optimal position"] = [sol]

                i = i+1
                print(df)

    # Write the DataFrame to a CSV file
    # df.to_csv(csv_file_path, index=True)  # Set index=True to include row index in the CSV

    print(f"DataFrame has been written to '{csv_file_path}'.")

    # Reset standard output to the terminal
    sys.stdout.log_file.close()
    sys.stdout = sys.__stdout__