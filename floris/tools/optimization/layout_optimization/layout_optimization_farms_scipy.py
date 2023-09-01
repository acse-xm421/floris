# Xuefei Mi acse-xm421
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
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Point

from .layout_optimization_scipy import LayoutOptimizationScipy
from .layout_optimization_farms_base import LayoutOptimizationFarmsBase



class LayoutOptimizationFarmsScipy(LayoutOptimizationFarmsBase, LayoutOptimizationScipy):
    def __init__(
        self,
        nfarms,
        fi_list,
        nturbs_list,
        angle_list,
        dist_list,
        boundary_1,
        min_dist=None,
        wind_directions=None,
        wind_speeds=None,
        freq=None,
        bnds=None,
        solver='SLSQP',
        chosen_weights="flexible",
        optOptions=None,
    ):
        """
        Initialize the LayoutOptimizationFarmsScipy class.

        Args:
            nfarms (_type_): _description_
            fi_list (list): List of farm instances (fi) for optimization.
            nturbs_list (list): List of the number of turbines in each farm.
            angle_list (list): List of angles for farm positioning. Every angle ranges between [0,360].
            dist_list (list): List of distances for farm positioning.
            boundary_1 (list): List of vertices defining the boundary of both farms.
                The boundary of the smallest square which can accomodate both farms.            
            min_dist (float, optional): Minimum distance to be maintained between turbines during optimization (m).
            wind_directions (list, optional): Specific wind directions ranging from [0,360].
                eg. [270.0, 300.0].
            wind_speeds (list, optional): Specific wind speeds. Need to be greater than 1. 
                eg. [8.0, 10.0, 12.0].
            freq (np.array, optional): Array of frequencies for wind conditions. Its shape needs
                to match the dimension of wind directions and wind speeds. Or. it will raise error.
                eg. freq.shape = (len(self.wind_directions), len(self.wind_speeds))
            bnds (iterable, optional): Bounds for the optimization
                variables (pairs of min/max values for each variable (m)). If
                none are specified, they are set to 0 and 1. Defaults to None.
            solver (str, optional): Sets the solver used by Scipy. Defaults to 'SLSQP'.
            chosen_weights (str): Weighting scheme for turbines. Here are three options:
                "fixed": fixed turbines have weight 1, flexible turbines have weight 0.
                "flexible": fixed turbines have weight 0, flexible turbines have weight 1.
                "both": fixed turbines have weight 1, flexible turbines have weight 1.
            optOptions (dict, optional): Dicitonary for setting the
                optimization options. Defaults to None.
        """
        # Call the initialization methods of the base classes to set up the attributes and constraints.
        super().__init__(nfarms, fi_list, nturbs_list, angle_list, dist_list, \
                         boundary_1, chosen_weights, min_dist, wind_directions, \
                            wind_speeds, freq)


        # normalize boundary_list
        self.boundary_list_norm = []
        for boundaries in self.boundary_list:
            boundary_norm = []
            for val in boundaries:
                boundary_tup = (
                    self._norm(val[0], self.farms_xmin, self.farms_xmax),
                    self._norm(val[1], self.farms_ymin, self.farms_ymax)
                )
                boundary_norm.append(boundary_tup)
            self.boundary_list_norm.append(boundary_norm)
        
        self.norm_x = [self._norm(x, self.farms_xmin, self.farms_xmax) for x in self.wf.layout_x]
        self.norm_y = [self._norm(y, self.farms_ymin, self.farms_ymax) for y in self.wf.layout_y]

        self.x0 = self.norm_x + self.norm_y #?

        # sure
        if bnds is not None:
            self.bnds = bnds
        else:
            self._set_opt_bounds()
        if solver is not None:
            self.solver = solver
        if optOptions is not None:
            self.optOptions = optOptions
        else:
            self.optOptions = {"maxiter": 50, "disp": True, "iprint": 2, "ftol": 1e-9, "eps":0.01}

        # set max, min
        self.xmin = self.farms_xmin
        self.xmax = self.farms_xmax
        self.ymin = self.farms_ymin
        self.ymax = self.farms_ymax

        self._generate_constraints()



    # Private method to set optimization bounds for turbine positions
    # Parameters:
    # - order: A list specifying the order of farms (default: ["fixed", "flexible"])
    # Returns:
    # - None
    def _set_opt_bounds(self, order=["fixed", "flexible"]):
        # Initialize lists to store bounds for x and y coordinates
        self.bnds_x = []
        self.bnds_y = []

        # Set bounds for each farm in the specified order
        for idx in range(self.nfarms):
            # Bounds for the fixed farm
            if idx == 0:
                for i in range(self.nturbs_list[idx]):
                    self.bnds_x.append((self.norm_x[i], self.norm_x[i]))
                    self.bnds_y.append((self.norm_y[i], self.norm_y[i]))
            # Bounds for the flexible farm
            else:
                # Normalize farm boundary coordinates
                bnd_xmin = self._norm(self.farm_xmin[idx], self.farms_xmin, self.farms_xmax)
                bnd_xmax = self._norm(self.farm_xmax[idx], self.farms_xmin, self.farms_xmax)
                bnd_ymin = self._norm(self.farm_ymin[idx], self.farms_ymin, self.farms_ymax)
                bnd_ymax = self._norm(self.farm_ymax[idx], self.farms_ymin, self.farms_ymax)

                # Set bounds for each turbine within the flexible farm
                self.bnds_x = self.bnds_x + [(bnd_xmin, bnd_xmax) for _ in range(self.nturbs_list[idx])]
                self.bnds_y = self.bnds_y + [(bnd_ymin, bnd_ymax) for _ in range(self.nturbs_list[idx])]

        # Combine x and y bounds to create overall bounds for optimization
        self.bnds = self.bnds_x + self.bnds_y

    # Private method to perform the optimization using a specified solver
    # Returns:
    # - The result of the optimization containing the optimal turbine positions
    def _optimize(self):
        # Perform the optimization using the minimize function
        self.residual_plant = minimize(
            self._obj_func,     # Objective function to be minimized
            self.x0,            # Initial guess for turbine positions
            method=self.solver, # Optimization solver method (e.g., "SLSQP", "L-BFGS-B")
            bounds=self.bnds,   # Bounds for turbine positions
            constraints=self.cons,  # Constraints (if any)
            options=self.optOptions,  # Optimization options (if any)
        )

        # Return the optimal turbine positions obtained from the optimization
        return self.residual_plant.x


    # Private method to calculate the objective function value for optimization
    # based on the given turbine positions (locs)
    # Args:
    # - locs (array-like): Array of turbine positions (normalized)
    # Returns:
    # - The negative of the farm's Annual Energy Production (AEP) divided by the initial AEP
    #   (A negative value is used because most optimization solvers aim to minimize)
    def _obj_func(self, locs):
        # Convert normalized turbine positions to unnormalized coordinates
        locs_unnorm = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        
        # Update the wind farm layout with the new turbine positions
        self._change_coordinates(locs_unnorm)
        
        # Calculate the negative of the farm's AEP divided by the initial AEP
        return -1 * self.wf.get_farm_AEP(freq=self.freq, turbine_weights=self.weights) / self.initial_AEP

    def _change_coordinates(self, locs):
        # Parse the layout coordinates
        layout_x = locs[0 : self.nturbs]
        layout_y = locs[self.nturbs : 2 * self.nturbs]

        # Update the turbine map in floris
        self.wf.reinitialize(layout_x=layout_x, layout_y=layout_y)

    def _generate_constraints(self):
        tmp1 = {
            "type": "ineq",
            "fun": lambda x, *args: self._space_constraint(x),
        }
        tmp2 = {
            "type": "ineq",
            "fun": lambda x: self._distance_from_boundaries(x),
        }

        self.cons = [tmp1]

    def _space_constraint(self, x_in, rho=500):
        x = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in x_in[0 : self.nturbs]
        ]
        y =  [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]

        # Calculate distances between turbines
        locs = np.vstack((x, y)).T
        distances = cdist(locs, locs)
        arange = np.arange(distances.shape[0])
        distances[arange, arange] = 1e10
        dist = np.min(distances, axis=0)

        g = 1 - np.array(dist) / self.min_dist

        # Following code copied from OpenMDAO KSComp().
        # Constraint is satisfied when KS_constraint <= 0
        g_max = np.max(np.atleast_2d(g), axis=-1)[:, np.newaxis]
        g_diff = g - g_max
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)[:, np.newaxis]
        KS_constraint = g_max + 1.0 / rho * np.log(summation)

        return -1*KS_constraint[0][0]

    # Private method to calculate the distances of turbines from flexible farm boundaries
    # Args:
    # - x_in (array-like): Array of turbine positions (normalized)
    # Returns:
    # - boundary_con (array): Array of distances from boundaries for each turbine
    #   (negative values indicate turbines outside the boundary, positive values inside)
    def _distance_from_boundaries(self, x_in):
        # Convert normalized turbine positions to unnormalized coordinates
        x = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in x_in[0 : self.nturbs]
        ]
        y =  [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]
        
        # Initialize an array to store distances from boundaries
        boundary_con = np.zeros(self.nturbs)
        
        # Consider turbines in the flexible farm (idx = 1)
        idx = 1
        for i in range(self.nturbs_list[idx]):
            loc = Point(x[i], y[i])
            # Calculate the distance of the turbine from the flexible farm boundary line
            boundary_con[i] = loc.distance(self._boundary_lines[idx])
            # If the turbine is inside the flexible farm boundary polygon, the distance is positive
            # Otherwise, it's negative to indicate being outside
            if self._boundary_polygons[idx].contains(loc) is True:
                boundary_con[i] *= 1.0
            else:
                boundary_con[i] *= -1.0

        return boundary_con

    def _get_initial_and_final_locs(self):
        x_initial = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in self.x0[0 : self.nturbs]
        ]
        y_initial = [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in self.x0[self.nturbs : 2 * self.nturbs]
        ]
        x_opt = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in self.residual_plant.x[0 : self.nturbs]
        ]
        y_opt = [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in self.residual_plant.x[self.nturbs : 2 * self.nturbs]#?
        ]
        return x_initial, y_initial, x_opt, y_opt

    # Public methods
    def optimize(self):
        """
        This method finds the optimized layout of wind turbines for power
        production given the provided frequencies of occurance of wind
        conditions (wind speed, direction).

        Returns:
            opt_locs (iterable): A list of the optimized locations of each
            turbine (m).
        """
        print("=====================================================")
        print("Optimizing turbine layout...")
        print("Number of parameters to optimize = ", len(self.x0))
        print("=====================================================")

        opt_locs_norm = self._optimize()

        print("Optimization complete.")

        opt_locs = [
            [
                self._unnorm(valx, self.farms_xmin, self.farms_xmax)
                for valx in opt_locs_norm[0 : self.nturbs]
            ],
            [
                self._unnorm(valy, self.farms_ymin, self.farms_ymax)
                for valy in opt_locs_norm[self.nturbs : 2 * self.nturbs]
            ],
        ]

        return opt_locs

