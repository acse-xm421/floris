import unittest
import numpy as np

# Import the LayoutOptimizationFarmsBase class (assuming it's in the same directory or properly installed)
from floris.tools.optimization.layout_optimization.layout_optimization_farms_scipy import (
    LayoutOptimizationFarmsBase
)
from floris.tools.optimization.layout_optimization.layout_optimization_scipy import LayoutOptimizationScipy
from floris.tools.optimization.layout_optimization.layout_optimization_farms_base import LayoutOptimizationFarmsBase
from floris.tools import FlorisInterface
from shapely.geometry import LineString, Polygon
from scipy.stats import norm


class TestLayoutOptimizationFarmsBase(unittest.TestCase):

    def setUp(self):
        # Create mock farm instances and other necessary inputs for testing
        nfarms = 2
        fi_1 = FlorisInterface("examples/inputs/gch.yaml") # New CumulativeCurl model
        fi_2 = FlorisInterface("examples/inputs/gch.yaml") # New CumulativeCurl model
        fi_list = [fi_1, fi_2]  # Replace with actual instances
        nturbs_list = [len(fi_1.layout_x), len(fi_2.layout_x)]
        angle_list = [30, 60]
        dist_list = [500, 800]
        boundary_1 = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]  # Example boundary
        chosen_weights = "flexible"
        min_dist = 200
        wind_directions = [0.0,]
        wind_speeds = [8.0,]
        freq = np.ones((len(wind_directions), len(wind_speeds))) / (len(wind_directions) * len(wind_speeds))

        # Create an instance of LayoutOptimizationFarmsBase for testing
        self.layout_optimizer = LayoutOptimizationFarmsBase(
            nfarms, fi_list, nturbs_list, angle_list, dist_list, boundary_1, chosen_weights, min_dist,
            wind_directions, wind_speeds, freq
        )

    def test_initialization(self):
        # Check if the layout optimizer is initialized properly
        self.assertEqual(self.layout_optimizer.nfarms, 2)
        self.assertEqual(self.layout_optimizer.chosen_weights, "flexible")
        self.assertEqual(self.layout_optimizer.min_dist, 200)

    def test_boundary_calculation(self):
        # Check if boundary calculation is performed correctly
        self.assertEqual(len(self.layout_optimizer.boundary_list), 2)
        self.assertEqual(len(self.layout_optimizer.dist_x), 2)
        self.assertEqual(len(self.layout_optimizer.dist_y), 2)
        self.assertEqual(len(self.layout_optimizer.farm_xmin), 2)
        self.assertEqual(len(self.layout_optimizer.farm_xmax), 2)
        self.assertEqual(len(self.layout_optimizer.farm_ymin), 2)
        self.assertEqual(len(self.layout_optimizer.farm_ymax), 2)
        self.assertTrue(isinstance(self.layout_optimizer._boundary_polygons[0], Polygon))
        self.assertTrue(isinstance(self.layout_optimizer._boundary_lines[0], LineString))

    def test_plot_layout_opt_results(self):
        # Check if the layout optimization results can be plotted without errors
        path = "test_plot.png"
        try:
            self.layout_optimizer.plot_layout_opt_results(path)
        except Exception as e:
            self.fail(f"Plotting layout optimization results failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
