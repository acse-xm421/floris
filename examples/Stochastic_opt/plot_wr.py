# Xuefei Mi acse-xm421
from floris.tools import WindRose
import matplotlib.pyplot as plt

# Read in the wind rose using the class
wind_rose = WindRose()
wind_rose.read_wind_rose_csv("examples/inputs/wind_rose.csv")

# fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))

# Show the wind rose
wind_rose.plot_wind_rose(legend_kwargs={'loc':'upper right', 'bbox_to_anchor': (1.3, 1.1)})
plt.savefig("examples/test_pic/wind_rose_loc.png")