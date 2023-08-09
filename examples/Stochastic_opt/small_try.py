import numpy as np
# wind_directions = np.arange(0, 360, 5.0)
# wind_speeds = [8.0]
# print(np.array(np.ones_like(wind_directions)/np.sum(np.ones_like(wind_directions))).reshape( ( len(wind_directions), len(wind_speeds) ) ))
# freq = (
#     np.array(np.ones_like(wind_directions)/np.sum(np.ones_like(wind_directions)))
#     .reshape( ( len(wind_directions), len(wind_speeds) ) )
# )
# print(freq)
# freq = freq / freq.sum()
# def dist(a, b):
#     return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# p1 = (115, 1)
# p2 = (523, 3)
# p3 = (358, 265)
# print('dist between p1 and p3', dist(p1, p3))
# print('dist between p2 and p3', dist(p2, p3))
# First party modules
from pyoptsparse import SLSQP, Optimization


# rst begin objfunc
def objfunc(xdict):
    x = xdict["xvars"]
    funcs = {}
    funcs["obj"] = -x[0] * x[1] * x[2]
    conval = [0] * 2
    conval[0] = x[0] + 2.0 * x[1] + 2.0 * x[2] - 72.0
    conval[1] = -x[0] - 2.0 * x[1] - 2.0 * x[2]
    funcs["con"] = conval
    fail = False

    return funcs, fail


# rst begin optProb
# Optimization Object
optProb = Optimization("TP037 Constraint Problem", objfunc)

# rst begin addVar
# Design Variables
optProb.addVarGroup("xvars", 3, "c", lower=[0, 0, 0], upper=[42, 42, 42], value=10)

# rst begin addCon
# Constraints
optProb.addConGroup("con", 2, lower=None, upper=0.0)

# rst begin addObj
# Objective
optProb.addObj("obj")

# rst begin print
# Check optimization problem
# print(optProb)

# rst begin OPT
# Optimizer
optOptions = {"IPRINT": 1}
opt = SLSQP(options=optOptions)

# rst begin solve
# Solve
sol = opt(optProb, sens="FD")
# Get optimized variable values
optimal_variable_values = sol.getDVs()["xvars"]

print(optimal_variable_values)
# rst begin check
# Check Solution
# print(sol)