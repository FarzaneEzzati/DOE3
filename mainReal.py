"""
This is the main code to build models and solve the forecast problems.
The results are saved to feed into the real time operation models.
The inputs are PV and Load forecast scenarios + various device specifications + penalties
The output is the GUROBI model which is saved as .mps file
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
from buildCEMS import buildForCEMS, buildRealCEMS
from buildMG import buildForMG, buildRealMG

import pickle as pkl
import warnings
warnings.filterwarnings('ignore')


class mg:
    """ we need this class to preserve MGs information all the time."""
    def __init__(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as handle:
            self.gamma, self.dv, self.ps, self.pv_s, self.l_s, self.effi = pkl.load(handle)
        handle.close()


if __name__ == '__main__':
    # Upload the information for the target microgrid
    mg0 = mg('pkls/mg0.pkl')
    mg1 = mg('pkls/mg1.pkl')
    mg2 = mg('pkls/mg2.pkl')

    # We build the models for each microgrid separately
    f_mg1 = buildForMG(mg1)
    f_mg1.optimize()
    Z_optimal = {t: f_mg1.getVarByName(f'Z{[t]}').x for t in range(6)}
    print(Z_optimal)

    # Assume the central EMS solved its model and V variables are assigned to each MG. Assume we have it for micorgrid 1.
    V1 = [30, 40, 20, 10, 50, 60]






    """ This section is for real time """
    # First we build the model for real-time operation of a MG then we feed inputs and solve sequentially.
    r_mg1 = buildRealMG(mg1)

    # Initialize the first value of inputs.
    # For E, we assume ES is full. For PV, we give the real forecast of PV generation. For load, we do the same thing.
    Ei = mg1.dv['es']  # E input to the model
    objs = []  # Save objective values (penalty of deviations).

    for t in range(6):
        PVi = mg1.pv_s[1][t] * 1.2  # Set PV input for the time t.
        Li = mg1.l_s[1][t] * 1.1  # Set L input for the time t.
        Zi = Z_optimal[t]  # Set Z input for the time t.
        Vi = V1[t]  # Set V as assigned value to help at time t.

        # Update RHS of the corresponding constraints.
        r_mg1.setAttr('RHS', r_mg1.getConstrByName('E level'), Ei)
        r_mg1.setAttr('RHS', r_mg1.getConstrByName('PV power'), PVi)
        r_mg1.setAttr('RHS', r_mg1.getConstrByName('Load components'), Li)
        r_mg1.setAttr('RHS', r_mg1.getConstrByName('abs'), Zi)
        r_mg1.setAttr('RHS', r_mg1.getConstrByName('V components'), Vi)

        # Now optimize the model.
        r_mg1.optimize()

        # Get the objective value
        objs.append(r_mg1.ObjVal)

        # Get E for the next period.
        Ei = r_mg1.getVarByName('E').x

        r_mg1.update()
    print(objs)





