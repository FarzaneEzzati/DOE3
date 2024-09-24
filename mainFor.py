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
from buildCEMS import buildForCEMS, buildRealCEMS  # import functions to build models for individual MGs
from buildMG import buildForMG, buildRealMG  # import functions to build models for CEMS
import pickle as pkl
import warnings
warnings.filterwarnings('ignore')


class mg_c:
    """ we need this class to preserve MGs information all the time."""
    def __init__(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as handle:
            self.gamma, self.dv, self.ps, self.pv_s, self.l_s, self.l_max, self.effi = pkl.load(handle)
        handle.close()


class cems_c:
    """We need this class to preserve CEMS information all the time."""
    def __init__(self, pkl_file_name):
        with open(pkl_file_name, 'rb') as handle:
            self.gamma, self.dv, self.ps, self.pv_s, self.effi = pkl.load(handle)
        handle.close()


if __name__ == '__main__':
    # Upload the information for the target microgrids (named mgi: microgrid information).
    mgi = {0: mg_c('pkls/mg0.pkl'), 1: mg_c('pkls/mg1.pkl'), 2: mg_c('pkls/mg2.pkl')}

    # We build the day-ahead models for each microgrid separately (named mgm: microgrid models).
    # Save the optimal solution as well in a dictionary (named mgs: microgrid solutions).
    mgm = {}
    mg_z, mg_zdg, mg_lmax, mg_ymax = {}, {}, {}, {}
    for key in mgi.keys():
        mgm[key] = buildForMG(mgi[key])
        mgm[key].optimize()

        # Get Z and Z_dg from obtained solutions.
        mg_z[key] = {t: mgm[key].getVarByName(f'Z[{t}]').x for t in range(6)}
        mg_zdg[key] = {t: mgm[key].getVarByName(f'Zdg[{t}]').x for t in range(6)}

        # Get maximum load shedding from solutions and load amount from information available.
        mg_lmax[key] = mgi[key].l_max
        mg_ymax[key] = {t: max([mgm[key].getVarByName(f'Y[{t},{s}]').x for s in range(4)]) for t in range(6)}

    # Upload information for the CEMS.
    cems = cems_c('pkls/cems.pkl')

    # Build the day-ahead model for CEMS given the forecast decision for ind. MGs.


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





