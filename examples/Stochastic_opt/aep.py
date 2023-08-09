# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


import numpy as np
import pandas as pd
from floris.tools import FlorisInterface


"""
This example creates a FLORIS instance
1) Makes a two-turbine layout
2) Demonstrates single ws/wd simulations
3) Demonstrates mulitple ws/wd simulations

Main concept is introduce FLORIS and illustrate essential structure of most-used FLORIS calls
"""

# Initialize FLORIS with the given input file via FlorisInterface.
# For basic usage, FlorisInterface provides a simplified and expressive
# entry point to the simulation routines.
fi = FlorisInterface("examples/inputs/gch.yaml")

# Convert to a simple two turbine layout
layout_x = [0, 0, 500.0, 500.0, 1000.0, 1000.0, ] # 3*D, 6 * D, 6 * D,
layout_y = [0, 1000.0, 0, 1000.0, 0, 1000.0,] # 4 * D, 0, 4 * D,
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)

# Get the turbine powers assuming 1 wind speed and 1 wind direction
wind_directions = np.arange(0, 360, 5.0)
wind_directions_data = {'wind_direction': wind_directions}
df = pd.DataFrame(wind_directions_data)

wind_speeds = [8.0]
freq = (
    np.array(np.ones_like(wind_directions)/np.sum(np.ones_like(wind_directions)))
    .reshape( ( len(wind_directions), len(wind_speeds) ) )
)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds)

# Set the yaw angles to 0
yaw_angles = np.array([[np.zeros(len(fi.layout_x))]])
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers_0 = fi.get_turbine_powers()
farm_power_0 = fi.get_farm_power()
base_aep_0 = fi.get_farm_AEP(freq=freq) / 1e6
print("turbine_powers: ", turbine_powers_0)
print("farm_power: ", farm_power_0)
print(f'AEP: {base_aep_0}')
df['farm_power_0'] = farm_power_0

layout_x = [0, 0, 500.0, ] # 3*D, 6 * D, 6 * D,
layout_y = [0, 1000.0, 1000.0, ] # 4 * D, 0, 4 * D,
fi.reinitialize(layout_x=layout_x, layout_y=layout_y)
fi.calculate_wake(yaw_angles=yaw_angles)
turbine_powers_1 = fi.get_turbine_powers()
farm_power_1 = fi.get_farm_power()
base_aep_1 = fi.get_farm_AEP(freq=freq) / 1e6
print("turbine_powers: ", turbine_powers_1)
print("farm_power: ", farm_power_1)
print(f'AEP: {base_aep_1}')
df['farm_power_1'] = farm_power_1

turbine_diff = turbine_powers_1 - turbine_powers_0
farm_diff = farm_power_1 - farm_power_0
aep_diff = base_aep_1 - base_aep_0

turbine_diff_reshaped = turbine_diff.reshape(72, 3)

turbine_diff_reshaped = pd.DataFrame(turbine_diff_reshaped, columns=['Turbine1', 'Turbine2', 'Turbine3'])

print("turbine_diff: ", turbine_diff_reshaped)
print("farm_diff: ", pd.DataFrame(farm_diff))
print("aep_diff: ", aep_diff)
df['farm_power_diff'] = farm_diff
df = pd.concat([df, turbine_diff_reshaped], axis=1)
print(df)
