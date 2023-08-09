import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def load_floris():
    # Load the default example floris object
    fi = FlorisInterface("examples/inputs/gch.yaml") # GCH model matched to the default "legacy_gauss" of V2
    # fi = FlorisInterface("inputs/cc.yaml") # New CumulativeCurl model

    # Specify the full wind farm layout: nominal wind farms
    # Set turbine locations to 6 turbines in a rectangle
    D = 126.0 # rotor diameter for the NREL 5MW
    layout_x = [0, 0, 6 * D, 6 * D, 3*D, 3*D,] #
    layout_y = [0, 4 * D, 0, 4 * D, 0, 4 * D,] #
    
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

if __name__ == "__main__":
    # Load FLORIS: full farm including neighboring wind farms
    fi, turbine_weights = load_floris()
    nturbs = len(fi.layout_x)

    # Load a dataframe containing the wind rose information
    ws_windrose, wd_windrose, freq_windrose = load_windrose()
    ws_windrose = ws_windrose + 0.001  # Deal with 0.0 m/s discrepancy

    # Create a FLORIS object for AEP calculations
    fi_WR = fi.copy()
    fi_WR.reinitialize(wind_speeds=ws_windrose, wind_directions=wd_windrose)
    # fi_WR.reinitialize(wind_speeds=[8.0], wind_directions=np.arange(0, 360, 5.0))

    # The boundaries for the turbines, specified as vertices
    boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]
    yaw_angles = np.array([[np.zeros(nturbs)]])
    show_wd = 90

    plot_horizontal(fi_WR, show_wd, "before_opt", yaw_angles=yaw_angles)


    # Setup the optimization problem
    layout_opt = LayoutOptimizationScipy(fi_WR, boundaries, freq=freq_windrose, min_dist=300)

    # Run the optimization
    sol = layout_opt.optimize()

    # Get the resulting improvement in AEP
    print('... calcuating improvement in AEP')
    fi_WR.calculate_wake()
    base_aep = fi_WR.get_farm_AEP(freq=freq_windrose, turbine_weights=turbine_weights) / 1e6
    fi_WR.reinitialize(layout_x=sol[0], layout_y=sol[1])

    plot_horizontal(fi_WR, show_wd, "after_opt", yaw_angles=yaw_angles)

    fi_WR.calculate_wake()
    opt_aep = fi_WR.get_farm_AEP(freq=freq_windrose, turbine_weights=turbine_weights) / 1e6
    percent_gain = 100 * (opt_aep - base_aep) / base_aep

    # Print and plot the results
    print(f'Optimal layout: {sol}')
    print(
        f'Optimal layout improves AEP by {percent_gain:.1f}% '
        f'from {base_aep:.1f} MWh to {opt_aep:.1f} MWh'
    )
    layout_opt.plot_layout_opt_results()
    plt.savefig("examples/test_pic/plot_stochastic.png")

