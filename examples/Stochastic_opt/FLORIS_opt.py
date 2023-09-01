# Xuefei Mi acse-xm421
import os
import matplotlib.pyplot as plt
import numpy as np

import floris.tools.visualization as wakeviz
from floris.tools import FlorisInterface
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

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('inputs/gch.yaml')

# Setup 1 wind directions with 1 wind speed and frequency distribution
wind_directions = [0.0,]
wind_speeds = [8.0,]
# Shape frequency distribution to match number of wind directions and wind speeds
freq = (
    np.abs(
        np.sort(
            np.random.randn(len(wind_directions))
        )
    )
    .reshape( ( len(wind_directions), len(wind_speeds) ) )
)
freq = freq / freq.sum()
print(freq)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# The boundaries for the turbines, specified as vertices
boundaries = [(0.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0), (1000.0, 0.0), (0.0, 0.0)]

# Set turbine locations to 4 turbines in a rectangle
D = 126.0 # rotor diameter for the NREL 5MW
layout_x = [0, 0, 3*D, 3*D, 6 * D, 6 * D, ]
layout_y = [0, 4 * D, 0, 4 * D, 0, 4 * D,]
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Create the plots of un-optimized layout
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
    yaw_angles=np.array([[[0.,0.,0.,0.,0.,0.]]]),
)
fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list, title="Horizontal")
plt.savefig("test_pic/plot_floris_01.png")

# Setup the optimization problem
layout_opt = LayoutOptimizationScipy(fi, boundaries, freq=freq)

# Run the optimization
sol = layout_opt.optimize()

# Get the resulting improvement in AEP
print('... calcuating improvement in AEP')
fi.calculate_wake()
base_aep = fi.get_farm_AEP(freq=freq) / 1e6
fi.reinitialize(layout_x=sol[0], layout_y=sol[1])

# Create the plot of optimized layout
horizontal_plane = fi.calculate_horizontal_plane(
    x_resolution=200,
    y_resolution=100,
    height=90.0,
    yaw_angles=np.array([[[0.,0.,0.,0.,0.,0.]]]),
)
fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
wakeviz.visualize_cut_plane(horizontal_plane, ax=ax_list, title="Horizontal")
plt.savefig("test_pic/plot_floris_02.png")

#calculate AEP
fi.calculate_wake()
opt_aep = fi.get_farm_AEP(freq=freq) / 1e6
percent_gain = 100 * (opt_aep - base_aep) / base_aep

# Print and plot the results
print(f'Optimal layout: {sol}')
print(
    f'Optimal layout improves AEP by {percent_gain:.1f}% '
    f'from {base_aep:.1f} MWh to {opt_aep:.1f} MWh'
)
layout_opt.plot_layout_opt_results()
plt.savefig("test_pic/plot_floris_03.png")