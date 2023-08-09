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

import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from shapely.geometry import Point
from shapely.geometry import LineString, Polygon

from .layout_optimization_scipy import LayoutOptimizationScipy
from .layout_optimization_base import LayoutOptimization



class LayoutOptimizationFarms(LayoutOptimization):
    def __init__(
        self,
        nfarms,
        fi_list,
        nturbs_list,
        angle_list,
        dist_list,
        boundary_1,
        bnds=None,
        freq=None,
        min_dist=None,
        solver='SLSQP',
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
        for idx, fi in enumerate(fi_list):
            farm1 = super().__init__(fi, boundary_1, min_dist=None, freq=None)

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

        self._calc_boundary()
        self.whole_boundaries = [(self.farms_xmin, self.farms_ymin),(self.farms_xmin, self.farms_ymax),\
                                 (self.farms_xmax, self.farms_ymax),(self.farms_xmax, self.farms_ymin),\
                                 (self.farms_xmin, self.farms_ymin)]
        self.nturbs = np.sum(nturbs_list)#see wf? checked

        # sure
        for idx, fi in enumerate(fi_list):
            if idx == 0:
                self.wf = fi.copy()
            else:
                self.wf._add_farm(idx, fi)

        self.initial_AEP = self.wf.get_farm_AEP(self.freq)
        
        self.x0 = [
            self._norm(x, self.farms_xmin, self.farms_xmax)
            for x in self.wf.layout_x
        ] + [
            self._norm(y, self.farms_ymin, self.farms_ymax)
            for y in self.wf.layout_y
        ]

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
            self.optOptions = {"maxiter": 100, "disp": True, "iprint": 2, "ftol": 1e-9, "eps":0.01}

        # not sure
        self._generate_constraints()


    # Private methods
    def _calc_boundary(self):
        self.boundary_list = []
        self.farm_xmin =[]
        self.farm_xmax =[]
        self.farm_ymin =[]
        self.farm_ymax =[]
        self._boundary_polygon = []#Polygon(self.boundaries)
        self._boundary_line = []#LineString(self.boundaries)
        self.dist_x = []
        self.dist_y = []

        # calculate boundary_list, _boundary_polygon, _boundary_line
        for idx, (angle, dist) in enumerate(tuple(self.angle_list, self.dist_list)):
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

            self._boundary_polygon.append(Polygon(self.boundary_list[idx]))
            self._boundary_line.append(LineString(self.boundary_list[idx]))
        
        self.farms_xmin = np.min(self.farm_xmin)
        self.farms_xmax = np.max(self.farm_xmax)
        self.farms_ymin = np.min(self.farm_ymin)
        self.farms_ymax = np.max(self.farm_ymax)

        # normalize boundary_list
        self.boundary_list_norm = []
        for boundaries in self.boundary_list:
            boundary_norm = []
            for val in boundaries:
                boundary_tup = []
                boundary_tup.append(self._norm(val[0], self.farms_xmin, self.farms_xmax))
                boundary_tup.append(self._norm(val[1], self.farms_ymin, self.farms_ymax))
                boundary_norm.append(boundary_tup)
            self.boundary_list_norm.append(boundary_norm)

    def _add_farm(self, idx, fi):

        layout_x = fi.layout_x + self.dist_x[idx]
        layout_y = fi.layout_y + self.dist_y[idx]
        layout_x_all = np.append(self.wf.layout_x, layout_x)
        layout_y_all = np.append(self.wf.layout_y, layout_y)
        self.wf.reinitialize(layout_x = layout_x_all, layout_y = layout_y_all)

    def _set_opt_bounds(self):
        self.bnds_x = []
        self.bnds_y = []

        # set bnds for each farm
        for idx in range(self.nfarms):
            bnd_xmin = self.farm_xmin[idx]/self.farms_xmax
            bnd_xmax = self.farm_xmax[idx]/self.farms_xmax
            bnd_ymin = self.farm_ymin[idx]/self.farms_ymax
            bnd_ymax = self.farm_ymax[idx]/self.farms_ymax

            # set bnds for each turbine
            self.bnds_x.append([(bnd_xmin, bnd_xmax) for _ in range(self.nturbs_list[idx])])
            self.bnds_y.append([(bnd_ymin, bnd_ymax) for _ in range(self.nturbs_list[idx])])

        self.bnds = self.bnds_x + self.bnds_y

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

    def _obj_func(self, locs):
        locs_unnorm = [
            self._unnorm(valx, self.farms_xmin, self.farms_xmax)
            for valx in locs[0 : self.nturbs]
        ] + [
            self._unnorm(valy, self.farms_ymin, self.farms_ymax)
            for valy in locs[self.nturbs : 2 * self.nturbs]
        ]
        self._change_coordinates(locs_unnorm)
        return -1 * self.wf.get_farm_AEP(self.freq) / self.initial_AEP

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

        self.cons = [tmp1, tmp2]

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
        for idx in range(self.nfarms):
            for i in range(self.nturb_list[idx]):
                loc = Point(x[i], y[i])
                boundary_con[i] = loc.distance(self._boundary_line[idx])# in base
                if self._boundary_polygon[idx].contains(loc) is True:
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
