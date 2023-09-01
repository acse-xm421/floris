# Xuefei Mi acse-xm421
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
from scipy.interpolate import NearestNDInterpolator
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import (
    LayoutOptimizationScipy,
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
distance = 0
length = 2000

def load_floris():
    # Load the default example floris object
    fi = FlorisInterface("examples/inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
    # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

    # Specify the full wind farm layout: nominal wind farms
    # Set turbine locations to 6 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    x = np.linspace(distance, distance+length, 4)
    y = np.linspace(distance, distance+length, 3)
    layout_x = [x[0], x[1], x[2], x[3], x[1], x[2], x[0], x[1], x[2], x[3]] # 3*D, 6 * D, 6 * D,
    layout_y = [y[0], y[0], y[0], y[0], y[1], y[1], y[2], y[2], y[2], y[2]] # 4 * D, 0, 4 * D,
    
    # Turbine weights: we want to only optimize for the first 10 turbines, can we? others fixed?
    turbine_weights = np.zeros(len(layout_x), dtype=int)
    turbine_weights[:] = 1.0

    # Now reinitialize FLORIS layout
    fi.reinitialize(layout_x = layout_x, layout_y = layout_y)

    # # And visualize the floris layout
    # fig, ax = plt.subplots()
    # ax.plot(X[turbine_weights == 0], Y[turbine_weights == 0], 'ro', label="Neighboring farms")
    # ax.plot(X[turbine_weights == 1], Y[turbine_weights == 1], 'go', label='Farm subset')
    # ax.grid(True)
    # ax.set_xlabel("x coordinate (m)")
    # ax.set_ylabel("y coordinate (m)")
    # ax.legend()

    return fi, turbine_weights

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

def plot_horizontal(fi, wd, save_name, yaw_angles):
    # Create the plots of un-optimized layout
    horizontal_plane = fi.calculate_horizontal_plane(
        x_resolution=200,
        y_resolution=100,
        height=90.0,
        yaw_angles=yaw_angles,
        wd=[wd],
        ws=[8.0],
    )
    fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
    wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list, title="Horizontal")
    name = "examples/test_pic/" + save_name + "_" + str(wd) + ".png"
    plt.savefig(name)

# def plot_layout(x_1, y_1,x_2,y_2,x_3,y_3, path, boundaries):
#     # x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

#     plt.figure(figsize=(9, 6))
#     fontsize = 14
#     plt.plot(x_1, y_1, "or")
#     # plt.plot(x_opt, y_opt, "or")
#     # plt.title('Layout Optimization Results', fontsize=fontsize)
#     plt.xlabel("x (m)", fontsize=fontsize)
#     plt.ylabel("y (m)", fontsize=fontsize)
#     plt.axis("equal")
#     plt.grid()
#     plt.tick_params(which="both", labelsize=fontsize)
#     plt.legend(
#         ["Turbine Positions"],
#         loc="lower center",
#         bbox_to_anchor=(0.5, 1.01),
#         ncol=2,
#         fontsize=fontsize,
#     )

#     verts = boundaries
#     for i in range(len(verts)):
#         if i == len(verts) - 1:
#             plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
#         else:
#             plt.plot(
#                 [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
#             )

#     plt.savefig(path)

import matplotlib.pyplot as plt

def plot_layout(x_1, y_1, x_2, y_2, x_3, y_3, path, boundaries):
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fontsize = 14

    # Plot for x_1
    axes[0].plot(x_1, y_1, "or")
    axes[0].set_xlabel("x (m)", fontsize=fontsize)
    axes[0].set_ylabel("y (m)", fontsize=fontsize)
    axes[0].set_title("(a)", fontsize=fontsize)
    axes[0].axis("equal")
    axes[0].grid()
    axes[0].tick_params(which="both", labelsize=fontsize)
    axes[0].legend(
        ["Baseline Turbine Positions"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        fontsize=fontsize,
    )
    # Plot for x_2
    axes[1].plot(x_2, y_2, "or")
    axes[1].set_xlabel("x (m)", fontsize=fontsize)
    axes[1].set_ylabel("y (m)", fontsize=fontsize)
    axes[1].set_title("(b)", fontsize=fontsize)
    axes[1].axis("equal")
    axes[1].grid()
    axes[1].tick_params(which="both", labelsize=fontsize)
    axes[1].legend(
        ["Deterministic Turbine Positions"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        fontsize=fontsize,
    )

    # Plot for x_3
    axes[2].plot(x_3, y_3, "or")
    axes[2].set_xlabel("x (m)", fontsize=fontsize)
    axes[2].set_ylabel("y (m)", fontsize=fontsize)
    axes[2].axis("equal")
    axes[2].grid()
    axes[2].tick_params(which="both", labelsize=fontsize)
    axes[2].legend(
        ["Initial Turbine Positions for OUU"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.26),
        ncol=2,
        fontsize=fontsize,
    )
    axes[2].set_title("(c)", fontsize=fontsize)


    # Draw the boundary lines
    verts = boundaries
    for idx in range(3):
        for i in range(len(verts)):
            if i == len(verts) - 1:
                axes[idx].plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                axes[idx].plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b")

    # Adjust layout
    plt.tight_layout()

    # Save the figure as a high-quality image (e.g., PDF) for use in your thesis
    plt.savefig(path, dpi=300, bbox_inches='tight')

# Call the function with your data and file path
# plot_layout(x_1, y_1, x_2, y_2, x_3, y_3, 'layout_plots.png', boundaries)

if __name__ == "__main__":
    # Load FLORIS: full farm including neighboring wind farms
    fi, turbine_weights = load_floris()
    nturbs = len(fi.layout_x)

    # Load a dataframe containing the wind rose information
    # ws_windrose, wd_windrose, freq_windrose = load_windrose()
    # ws_windrose = ws_windrose + 0.001  # Deal with 0.0 m/s discrepancy

    wind_directions = np.arange(-180, 180, 5.0) #?
    wind_speeds = [8.0]

    variance_wd = 15.0
    mean_wd = 0.0

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
    fi_WR.reinitialize(wind_speeds=wind_speeds, wind_directions=wind_directions)
    # fi_WR.reinitialize(wind_speeds=[8.0], wind_directions=np.arange(0, 360, 5.0))

    # The boundaries for the turbines, specified as vertices
    boundaries = [(0.0, 0.0), (0.0, 2000.0), (2000.0, 2000.0), (2000.0, 0.0), (0.0, 0.0)]
    yaw_angles = np.array([[np.zeros(nturbs)]])
    show_wd = 0

    plot_horizontal(fi_WR, show_wd, "before_opt", yaw_angles=yaw_angles)

    x = np.linspace(distance, distance+length, 4)
    y = np.linspace(distance, distance+length, 3)
    x_1 = [x[0], x[1], x[2], x[3], x[1], x[2], x[0], x[1], x[2], x[3]] # 3*D, 6 * D, 6 * D,
    y_1 = [y[0], y[0], y[0], y[0], y[1], y[1], y[2], y[2], y[2], y[2]] # 4 * D, 0, 4 * D,

    x_2 = np.linspace(distance,distance+length,10) # 3*D, 6 * D, 6 * D,
    y_2 = np.repeat(distance+0.5*length, 10) #np.random.randint(distance, distance+length+1, size=10) # 4 * D, 0, 4 * D,
 
    x_3 = np.linspace(distance,distance+length,10) # 3*D, 6 * D, 6 * D,
    y_3 = np.random.randint(distance, distance+length+1, size=10) # 4 * D, 0, 4 * D,


    plot_layout(x_1, y_1, x_2, y_2, x_3, y_3, "examples/test_pic/plot_initial.png", boundaries)
    # plot_layout(fi.layout_x, fi.layout_y, "examples/test_pic/plot_initial.png", boundaries)

    # # Setup the optimization problem
    # layout_opt = LayoutOptimizationScipy(fi_WR, boundaries, freq=freq_windrose, min_dist=300)

    # # Run the optimization
    # sol = layout_opt.optimize()

    # # Get the resulting improvement in AEP
    # print('... calcuating improvement in AEP')
    # fi_WR.calculate_wake()
    fi_WR.reinitialize(layout_x=x_1, layout_y=y_1)
    base_aep_1 = fi_WR.get_farm_AEP(freq=freq, turbine_weights=turbine_weights) / 1e6
    print(f'1 AEP: {base_aep_1:.1f} MWh')

    x_1 = [x[0], x[1], x[2], x[3], x[0], x[3], x[0], x[1], x[2], x[3]] # 3*D, 6 * D, 6 * D,
    y_1 = [y[0], y[0], y[0], y[0], y[1], y[1], y[2], y[2], y[2], y[2]] # 4 * D, 0, 4 * D,

    fi_WR.reinitialize(layout_x=x_1, layout_y=y_1)
    base_aep_2 = fi_WR.get_farm_AEP(freq=freq, turbine_weights=turbine_weights) / 1e6
    print(f'2 AEP: {base_aep_2:.1f} MWh')

    diff = ((base_aep_1 - base_aep_2)/base_aep_1)*100
    print(f'AEP diff: {diff} ')


    # fi_WR.reinitialize(layout_x=sol[0], layout_y=sol[1])

    # plot_horizontal(fi_WR, show_wd, "after_opt", yaw_angles=yaw_angles)

    # fi_WR.calculate_wake()
    # opt_aep = fi_WR.get_farm_AEP(freq=freq_windrose, turbine_weights=turbine_weights) / 1e6
    # percent_gain = 100 * (opt_aep - base_aep) / base_aep

    # # Print and plot the results
    # print(f'Optimal layout: {sol}')
    # print(
    #     f'Optimal layout improves AEP by {percent_gain:.1f}% '
    #     f'from {base_aep:.1f} MWh to {opt_aep:.1f} MWh'
    # )
    # layout_opt.plot_layout_opt_results()
    # plt.savefig("examples/test_pic/plot_stochastic.png")

