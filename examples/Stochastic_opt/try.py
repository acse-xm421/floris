import numpy as np
import matplotlib.pyplot as plt

from floris.tools import FlorisInterface

fi = FlorisInterface("examples/inputs/gch.yaml")
x, y = fi.get_turbine_layout()

print("     x       y")
for _x, _y in zip(x, y):
    print(f"{_x:6.1f}, {_y:6.1f}")

x_2x2 = [0, 0, 800, 800]
y_2x2 = [0, 400, 0, 400]
fi.reinitialize(layout_x=x_2x2, layout_y=y_2x2)
print("fi.nturbs", fi.floris.farm.n_turbines)
x, y = fi.get_turbine_layout()

print("     x       y")
for _x, _y in zip(x, y):
    print(f"{_x:6.1f}, {_y:6.1f}")

# One wind direction and one speed -> one atmospheric condition
fi.reinitialize(wind_directions=[270.0], wind_speeds=[8.0])

# Two wind directions and one speed -> two atmospheric conditions
fi.reinitialize(wind_directions=[270.0, 280.0], wind_speeds=[8.0])

# Two wind directions and two speeds -> four atmospheric conditions
fi.reinitialize(wind_directions=[270.0, 280.0], wind_speeds=[8.0, 9.0])

fi.calculate_wake()

powers = fi.get_turbine_powers() / 1000.0  # calculated in Watts, so convert to kW

print("Dimensions of `powers`")
print( np.shape(powers) )

N_TURBINES = fi.floris.farm.n_turbines

print()
print("Turbine powers for 8 m/s")
for i in range(2):
    print(f"Wind direction {i}")
    for j in range(N_TURBINES):
        print(f"  Turbine {j} - {powers[i, 0, j]:7,.2f} kW")
    print()

print("Turbine powers for all turbines at all wind conditions")
print(powers)

farm_power=fi.get_farm_power()
print("farm power")
print(farm_power)

freq=np.array([[0.3,0.4],[0.1,0.2]])
aep=fi.get_farm_AEP(freq=freq)
print("aep")
print(aep)