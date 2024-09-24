import gurobipy as gp
from gurobipy import GRB


def buildForCEMS(cems, z, zdg, lmax, ymax):
    """Build the day ahead scheduling model for CEMS using the information available from individual MGs"""
    # z: total bidding, zdg: bidding from DG, lmax: max load demand, ymax: maximum shed amount (all in dict. MG # key)

    # Find the time periods and count of MGs.
    T = len(z[0].keys())  # Number of time periods
    T_range = range(T)
    I = len(z.keys())  # Number of microgrids
    I_range = range(I)
    IT_range = [(i, t) for i in I_range for t in T_range]  # the list of indices for decisions variables in stage 2
    IJT_range = [(i, j, t) for i in I_range for j in I_range for t in T_range]
    model = gp.Model('ForCEMS')
    # Decision variables (forecast)
    V = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='V')  # V: allocated power at time t to MG i

    R = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='R')  # R: allocated power from shared resources at time t to MG i
    K = model.addVars(IJT_range, vtype=GRB.CONTINUOUS, name='K')  # K: power transitioned from MG i to j at time t

    Wes = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wes')  # Wes: power from ES shared resource at time t
    Wpv = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wpv')  # Wes: power from ES shared resource at time t
    Wdg = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wdg')  # Wes: power from ES shared resource at time t

    

def buildRealCEMS():
    pass