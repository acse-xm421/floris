# Copyright 2022 NREL

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
import matplotlib.pyplot as plt
import math
from shapely.geometry import LineString, Polygon

from .layout_optimization_base import LayoutOptimization


# inherit _norm, _unnorm, plot_layout_opt_results
class LayoutOptimizationFarmsBase(LayoutOptimization):
    def __init__(
        self,
        nfarms,
        fi_list,
        nturbs_list,
        angle_list,
        dist_list,
        boundary_1,
        chosen_weights,
        min_dist=None,
        wind_directions=None,
        wind_speeds=None,
        freq=None,
        # bnds=None,
        # solver='SLSQP',
        # optOptions=None,
    ):
        """
        _summary_

        Args:
            fi (_type_): _description_
            boundaries (iterable(float, float)): Pairs of x- and y-coordinates
                that represent the boundary's vertices (m).
            freq (np.array): An array of the frequencies of occurance
                correponding to each pair of wind direction and wind speed
                values. If None, equal weight is given to each pair of wind conditions
                Defaults to None.
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to 0 and 1. Defaults to None.
            min_dist (float, optional): The minimum distance to be maintained
                between turbines during the optimization (m). If not specified,
                initializes to 2 rotor diameters. Defaults to None.
            solver (str, optional): Sets the solver used by Scipy. Defaults to 'SLSQP'.
            optOptions (dict, optional): Dicitonary for setting the
                optimization options. Defaults to None.
        """
        # not sure
        # for idx, fi in enumerate(fi_list):
        #     def __init__(self, fi, boundaries, min_dist=None, freq=None)
        #     farm1 = super().__init__(fi, boundary_1, min_dist=None, freq=None)

        # sure
        # Check if the lengths match
        if len(fi_list) != nfarms or \
            len(nturbs_list) != nfarms or \
            len(angle_list) != nfarms or \
            len(dist_list) != nfarms:
            raise ValueError("The lengths of the lists do not match nfarms.")

        self.fi_list = fi_list
        self.nturbs_list = nturbs_list
        self.angle_list = angle_list
        self.dist_list = dist_list
        self.boundary_1 = boundary_1
        self.nfarms = nfarms
        self.chosen_weights = chosen_weights
        
        # If no minimum distance is provided, assume a value of 2 rotor diamters
        if min_dist is None:
            self.min_dist = 200
        else:
            self.min_dist = min_dist

        if wind_directions is None:
            self.wind_directions = fi_list[0].floris.flow_field.n_wind_directions
        else:
            self.wind_directions = wind_directions

        if wind_speeds is None:
            self.wind_speeds = fi_list[0].floris.flow_field.n_wind_speeds
        else:
            self.wind_speeds = wind_speeds

        # print("wind_directions: ", self.wind_directions)
        # print("wind_speeds: ", self.wind_speeds)

        # If freq is not provided, give equal weight to all wind conditions
        if freq is None:
            self.freq = np.ones((
                len(self.wind_directions),
                len(self.wind_speeds)
            ))
            self.freq = self.freq / self.freq.sum()
            # self.freq = (
            #     np.array(np.ones_like(wind_directions)/np.sum(np.ones_like(wind_directions)))
            #     .reshape( ( len(wind_directions), len(wind_speeds) ) )
            # )
        else:
            self.freq = freq

        # calculate and choose weights
        self._calc_weights()
        self.weights = self._choose_weights(chosen_weights)

        #seperate boundary
        self._calc_boundary()

        # whole boundary
        self.whole_boundaries = [(self.farms_xmin, self.farms_ymin),(self.farms_xmin, self.farms_ymax),\
                                 (self.farms_xmax, self.farms_ymax),(self.farms_xmax, self.farms_ymin),\
                                 (self.farms_xmin, self.farms_ymin)]
        self.boundaries = self.whole_boundaries
        self._boundary_polygon = Polygon(self.boundaries)
        self._boundary_line = LineString(self.boundaries)

        # sure
        for idx, fi in enumerate(fi_list):
            if idx == 0:
                self.wf = fi.copy()
            else:
                self._add_farm(idx, fi)

        self.wf.reinitialize(wind_directions=self.wind_directions, wind_speeds=self.wind_speeds)
        self.initial_AEP = self.wf.get_farm_AEP(freq=self.freq, turbine_weights=self.weights)
        print("chosen_weights: ", self.chosen_weights)
        print("Initial AEP: ", self.initial_AEP)
        
    # Private methods
    def _calc_boundary(self):
        self.boundary_list = []
        self.dist_x = []
        self.dist_y = []
        self.farm_xmin =[]
        self.farm_xmax =[]
        self.farm_ymin =[]
        self.farm_ymax =[]
        self._boundary_polygons = []
        self._boundary_lines = []


        # calculate boundary_list, _boundary_polygon, _boundary_line
        for idx, (angle, dist) in enumerate(zip(self.angle_list, self.dist_list)):
            # calculate x and y distance
            self.dist_x.append(dist*math.cos(math.radians(angle)))
            self.dist_y.append(dist*math.sin(math.radians(angle)))

            if idx == 0:
                xmin = np.min([tup[0] for tup in self.boundary_1])
                xmax = np.max([tup[0] for tup in self.boundary_1])
                ymin = np.min([tup[1] for tup in self.boundary_1])
                ymax = np.max([tup[1] for tup in self.boundary_1])
                self.farm_xmin.append(xmin)
                self.farm_xmax.append(xmax)
                self.farm_ymin.append(ymin)
                self.farm_ymax.append(ymax)

                self.boundary_list.append(self.boundary_1)

                # calculate length and width
                self.length = xmax - xmin
                self.width = ymax - ymin

            else:
                xmin = self.farm_xmin[0] + self.dist_x[idx]
                xmax = self.farm_xmin[0] + self.dist_x[idx] + self.length
                ymin = self.farm_ymin[0] + self.dist_y[idx]
                ymax = self.farm_ymin[0] + self.dist_y[idx] + self.width

                self.farm_xmin.append(xmin)
                self.farm_xmax.append(xmax)
                self.farm_ymin.append(ymin)
                self.farm_ymax.append(ymax)

                boundary = [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
                self.boundary_list.append(boundary)

            self._boundary_polygons.append(Polygon(self.boundary_list[idx]))
            self._boundary_lines.append(LineString(self.boundary_list[idx]))
        
        # calculate farms boundary
        self.farms_xmin = np.min(self.farm_xmin)
        self.farms_xmax = np.max(self.farm_xmax)
        self.farms_ymin = np.min(self.farm_ymin)
        self.farms_ymax = np.max(self.farm_ymax)

    def _add_farm(self, idx, fi):

        layout_x = fi.layout_x + self.dist_x[idx]
        layout_y = fi.layout_y + self.dist_y[idx]
        layout_x_all = np.append(self.wf.layout_x, layout_x)
        layout_y_all = np.append(self.wf.layout_y, layout_y)
        self.wf.reinitialize(layout_x = layout_x_all, layout_y = layout_y_all)

    def _calc_weights(self):
        turbine_weights_fixed_ones = np.ones(self.nturbs_list[0], dtype=int)
        turbine_weights_fixed_zeros = np.zeros(self.nturbs_list[0], dtype=int)
        turbine_weights_flexible_ones = np.ones(self.nturbs_list[1], dtype=int)
        turbine_weights_flexible_zeros = np.zeros(self.nturbs_list[1], dtype=int)

        # fixed, flexible
        self.fixed_farm_weights = np.concatenate((turbine_weights_fixed_ones, turbine_weights_flexible_zeros))
        self.flexible_farm_weights = np.concatenate((turbine_weights_fixed_zeros, turbine_weights_flexible_ones))
        self.both_farms_weights = np.concatenate((turbine_weights_fixed_ones, turbine_weights_flexible_ones))

    def _choose_weights(self, chosen_weights):
        # choose weights
        if chosen_weights == "fixed":
            weights = self.fixed_farm_weights
        elif chosen_weights == "flexible":
            weights = self.flexible_farm_weights
        elif chosen_weights == "both":
            weights = self.both_farms_weights
        else:
            raise ValueError("The chosen_weights is not valid.")
        
        return weights
    
    # def plot_boundary(self, ax, boundary_names, boundary_styles):
    #     # Plot the boundary
    #     for i in range(len(self.boundary_list)):
    #         x_coords, y_coords = zip(*self.boundary_list[i])
    #         ax.plot(x_coords, y_coords, linestyle=boundary_styles[i], label=boundary_names[i], color='k')
    #         ax.legend()

    def plot_layout_opt_results(self, path):
        x_initial, y_initial, x_opt, y_opt = self._get_initial_and_final_locs()

        fig, ax= plt.subplots(1, 1, figsize=(9, 6))

        fontsize = 16
        ax.plot(x_initial, y_initial, "ob")
        ax.plot(x_opt, y_opt, "or")

        # plt.title('Layout Optimization Results', fontsize=fontsize)
        ax.set_xlabel("x (m)", fontsize=fontsize)
        ax.set_ylabel("y (m)", fontsize=fontsize)
        # ax.set_title("Layout Optimization Results", fontsize=fontsize)
        # ax.xlabel("x (m)", fontsize=fontsize)
        # ax.ylabel("y (m)", fontsize=fontsize)
        ax.axis("equal")
        ax.grid()
        ax.tick_params(which="both", labelsize=fontsize)
        legend1 = ax.legend(
            ["Old locations", "New locations"],
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=2,
            fontsize=fontsize,
        )
        ax.add_artist(legend1)

        verts = self.boundaries
        for i in range(len(verts)):
            if i == len(verts) - 1:
                plt.plot([verts[i][0], verts[0][0]], [verts[i][1], verts[0][1]], "b")
            else:
                plt.plot(
                    [verts[i][0], verts[i + 1][0]], [verts[i][1], verts[i + 1][1]], "b"
                )

        # plot boundary
        boundary_styles = ["--", ":"]
        boundary_names = ["fixed", "flexible"]
        for i in range(len(self.boundary_list)):
            x_coords, y_coords = zip(*self.boundary_list[i])
            ax.plot(x_coords, y_coords, linestyle=boundary_styles[i], label=boundary_names[i], color='k')
            legend2 = ax.legend()

        ax.add_artist(legend2)

        plt.savefig(path)
        # plt.show()



    ###########################################################################
    # Properties
    ###########################################################################

    @property
    def nturbs(self):
        """
        This property returns the number of turbines in the FLORIS
        object.

        Returns:
            nturbs (int): The number of turbines in the FLORIS object.
        """
        self._nturbs = self.wf.floris.farm.n_turbines
        return self._nturbs

    @property
    def rotor_diameter(self):
        return self.wf.floris.farm.rotor_diameters_sorted[0][0][0]