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
                # boundary_tup = ()
                # boundary_tup.append(self._norm(val[0], self.farms_xmin, self.farms_xmax))
                # boundary_tup.append(self._norm(val[1], self.farms_ymin, self.farms_ymax))
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



    def _set_opt_bounds(self, order=["fixed", "flexible"]):
        self.bnds_x = []
        self.bnds_y = []

        # set bnds for each farm
        for idx in range(self.nfarms):
            # fixed farm bounds
            if idx == 0:
                for i in range(self.nturbs_list[idx]):
                    self.bnds_x.append((self.norm_x[i], self.norm_x[i]))
                    self.bnds_y.append((self.norm_y[i], self.norm_y[i]))
            # flexible farm bounds
            else:
                bnd_xmin = self._norm(self.farm_xmin[idx], self.farms_xmin, self.farms_xmax)
                bnd_xmax = self._norm(self.farm_xmax[idx], self.farms_xmin, self.farms_xmax)
                bnd_ymin = self._norm(self.farm_ymin[idx], self.farms_ymin, self.farms_ymax)
                bnd_ymax = self._norm(self.farm_ymax[idx], self.farms_ymin, self.farms_ymax)

                # print("bnd_xmin=", bnd_xmin)
                # print("bnd_xmax=", bnd_xmax)
                # print("bnd_ymin=", bnd_ymin)
                # print("bnd_ymax=", bnd_ymax)


                # set bnds for each turbine
                self.bnds_x = self.bnds_x + [(bnd_xmin, bnd_xmax) for _ in range(self.nturbs_list[idx])]
                self.bnds_y = self.bnds_y + [(bnd_ymin, bnd_ymax) for _ in range(self.nturbs_list[idx])]

        self.bnds = self.bnds_x + self.bnds_y
        # print(self.bnds)

    def _optimize(self):
        self.residual_plant = minimize(
            self._obj_func,# checked
            self.x0,#?checked
            method=self.solver,
            bounds=self.bnds,
            constraints=self.cons,#?
            options=self.optOptions,
        )

        return self.residual_plant.x

    # wf changed
    def _obj_func(self, locs):
        locs_unnorm = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        self._change_coordinates(locs_unnorm)
        # objective_value_farms =  # able to choose
        # print("objective_value_farms=", objective_value_farms)
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

    # not used yet
    def _distance_from_boundaries(self, x_in):
        x = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in x_in[0 : self.nturbs]
        ]
        y =  [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in x_in[self.nturbs : 2 * self.nturbs]
        ]
        boundary_con = np.zeros(self.nturbs)

        # only flexible
        # for idx in range(self.nfarms):
        idx = 1
        for i in range(self.nturbs_list[idx]):
            loc = Point(x[i], y[i])
            boundary_con[i] = loc.distance(self._boundary_lines[idx])# in base
            if self._boundary_polygons[idx].contains(loc) is True:
                boundary_con[i] *= 1.0
            else:
                boundary_con[i] *= -1.0

        

        # for i in range(self.nturbs):
        #     loc = Point(x[i], y[i])
        #     boundary_con[i] = loc.distance(self._boundary_line)# in base
        #     if self._boundary_polygon.contains(loc) is True:
        #         boundary_con[i] *= 1.0
        #     else:
        #         boundary_con[i] *= -1.0

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
