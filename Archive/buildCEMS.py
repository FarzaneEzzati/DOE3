import gurobipy as gp
from gurobipy import GRB


def buildDACEMS(cems, Z, Xdg, Lmax, Ymax, mgi, SH):
    """Build the day ahead scheduling model for CEMS using the information available from individual MGs"""
    # Z: total bidding, Xdg: bidding from DG, Lmax: max load demand, Ymax: maximum shed amount
    # alphas: social vulnerability value (all in dict. MG # key)

    # Find the time periods and count of MGs.
    epsilon = 0.0002
    T = SH  # Number of time periods
    T_range = range(T)
    T1_range = range(T+1)
    I = len(Z.keys())  # Number of microgrids
    I_range = range(I)
    S = len(cems.ps)
    S_range = range(S)
    IT_range = [(i, t) for i in I_range for t in T_range]  # the list of indices for decisions variables in stage 2
    TS_range = [(t, s) for t in T_range for s in S_range]
    T1S_range = [(t, s) for t in T1_range for s in S_range]
    IJT_range = [(i, j, t) for i in I_range for j in I_range for t in T_range]  # The list of indices for ijt
    IJTS_range = [(i, j, t, s) for i in I_range for j in I_range for t in T_range for s in S_range]  # The list of indices for ijt
    ITS_range = [(i, t, s) for i in I_range for t in T_range for s in S_range]  # The list of indices for its

    model = gp.Model('DACEMS')
    model.setParam('OutputFlag', 0)
    # Decision variables (stage 1)
    V = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='V')  # V: allocated power at time t to MG i
    Wes = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wes')  # Wes: power from ES shared resource at time t
    Wpv = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wpv')  # Wes: power from ES shared resource at time t
    Wdg = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wdg')  # Wes: power from ES shared resource at time t
    R = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='R')  # R: allocated power from shared resources at time t to MG i
    K = model.addVars(IJT_range, vtype=GRB.CONTINUOUS, name='K')  # K: power transitioned from MG i to j at time t

    # Decision variables (stage 2)
    Wpv_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wpv_es')  # Wpv_es: charging es with pv
    Wdg_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wdg_es')  # Wpv_es: charging es with pv

    V_ = model.addVars(ITS_range, vtype=GRB.CONTINUOUS, name='V_')  # Vs: defining the V allocation under scenario s
    Wes_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wes_')  # Wes_: power from ES shared resource under sce s
    Wpv_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wpv_')  # Wpv_: power from PV shared resource under sce s
    Wdg_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wdg_')  # Wdg_: power from DG shared resource under sce s
    R_ = model.addVars(ITS_range, vtype=GRB.CONTINUOUS, name='R_')  # R_: allocated power from shared resources under sce s
    K_ = model.addVars(IJTS_range, vtype=GRB.CONTINUOUS, name='K_')  # K_: power transitioned from MG i to j under sce s
    E_ = model.addVars(T1S_range, vtype=GRB.CONTINUOUS, name='E')  # E: ES level

    # Decision variables (model absolute values of Delta)
    V_a = model.addVars(ITS_range, vtype=GRB.CONTINUOUS, name='V_a')  # the objective term corresponding to a+b
    V_b = model.addVars(ITS_range, vtype=GRB.CONTINUOUS, name='V_b')
    Wes_a = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wes_a')
    Wes_b = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wes_b')
    Wpv_a = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wpv_a')
    Wpv_b = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wpv_b')
    Wdg_a = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wdg_a')
    Wdg_b = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Wdg_b')

    # Constraints (stage 1)
    model.addConstrs(V[i, t] == R[i, t] + sum(K[j, i, t] for j in I_range) for i in I_range for t in T_range)  # Power from other MGs and shared resources make V
    model.addConstrs(V[i, t] <= Ymax[i][t] for i in I_range for t in T_range)  # V must be smaller than or equal to Ymax
    model.addConstrs(sum(R[i, t] for i in I_range) == Wes[t] + Wpv[t] + Wdg[t] for t in T_range)  # Balance for shared resources
    model.addConstrs(sum(K[i, j, t] for j in I_range) <= Z[i][t] for i in I_range for t in T_range)  # Balance in MG outputs
    #  model.addConstrs(K[i, i, t] == 0 for i in I_range for t in T_range)  # Exchange to itself not allowed
    model.addConstrs(mgi[i].Klimits['lower'] - V[i, t] <= 0for i in I_range for t in T_range)  # Lower limit on power amount to be assigned to MG
    model.addConstrs(V[i, t] <= mgi[i].Klimits['upper'] for i in I_range for t in T_range)  # Upper limit on power amount to be

    # Shared resources limits
    model.addConstrs(Wpv[t] <= cems.dv['pv'] * cems.dv['pv'] for t in T_range)  # PV components
    model.addConstrs(Wdg[t] <= cems.effic['dg effic'] * cems.dv['dg'] for t in T_range)  # DG components
    model.addConstrs(Wes[t] <= cems.dv['es'] for t in T_range)  # DG components

    # Constraints (stage 2) Linearization
    model.addConstrs(V[i, t] - V_[i, t, s] == V_a[i, t, s] - V_b[i, t, s] for i in I_range for t in T_range for s in S_range)  # Linear model for V deviation
    model.addConstrs(Wes[t] - Wes_[t, s] == Wes_a[t, s] - Wes_b[t, s] for t in T_range for s in S_range)  # Linear model for ES deviation
    model.addConstrs(Wpv[t] - Wpv_[t, s] == Wpv_a[t, s] - Wpv_b[t, s] for t in T_range for s in S_range)  # Linear model for PV deviation
    model.addConstrs(Wdg[t] - Wdg_[t, s] == Wdg_a[t, s] - Wdg_b[t, s] for t in T_range for s in S_range)  # Linear model for DG deviation

    # Constraints (ES level model)
    model.addConstrs(V_[i, t, s] == R_[i, t, s] + sum(K_[j, i, t, s] for j in I_range) for i in I_range for t in T_range for s in S_range)  # Power from other MGs and shared resources make V
    model.addConstrs(V_[i, t, s] <= Ymax[i][t] for i in I_range for t in T_range for s in S_range)  # V must be smaller than or equal to Ymax
    model.addConstrs(E_[0, s] == cems.E0 * cems.dv['es'] for s in S_range)  # ES level full starting each day
    model.addConstrs(E_[t + 1, s] == E_[t, s] + cems.effic['es effic'] * (Wpv_es[t, s] + Wdg_es[t, s]) -
                     (Wes_[t, s]) / cems.effic['es effic'] for s in S_range for t in T_range)  # Flow of power in ES
    # ES level should follow ES capacity
    model.addConstrs(E_[ts] <= cems.dv['es'] for ts in T1S_range)  # E should not exceed es capacity
    model.addConstrs(sum(R_[i, t, s] for i in I_range) == Wes_[t, s] + Wpv_[t, s] + Wdg_[t, s] for t in T_range for s in S_range)  # Balance for shared resources
    model.addConstrs(sum(K_[i, j, t, s] for j in I_range) <= Z[i][t] for i in I_range for t in T_range for s in S_range)  # Balance in MG outputs
    #  model.addConstrs(K_[i, i, t, s] == 0 for i in I_range for t in T_range for s in S_range)  # Exchange to itself not allowed
    model.addConstrs(mgi[i].Klimits['lower'] - V_[i, t, s] <= 0 for i in I_range for t in T_range for s in S_range)  # Lower limit on power amount to be assigned to MG
    model.addConstrs(V_[i, t, s] <= mgi[i].Klimits['upper'] for i in I_range for t in T_range for s in S_range)  # Upper limit on power amount to be

    # Shared resources limits
    model.addConstrs(Wpv_[t, s] + Wpv_es[t, s] <= cems.dv['pv'] * cems.pv_s[s][t] for t in T_range for s in S_range)  # PV components
    model.addConstrs(Wdg_[t, s] + Wdg_es[t, s] <= cems.effic['dg effic'] * cems.dv['dg'] for t in T_range for s in S_range)  # DG components
    model.addConstrs(Wes_[t, s] <= cems.dv['es'] for t in T_range for s in S_range)

    # Expected deviation costs
    Qc = sum(cems.ps[its[2]] * (V_a[its] + V_b[its])/(epsilon + Ymax[its[0]][its[1]]) for its in ITS_range) +\
         sum(cems.ps[ts[1]] * ((Wes_a[ts] + Wes_b[ts])/(epsilon + cems.dv['es']) +
                               (Wpv_a[ts] + Wpv_b[ts])/(epsilon + cems.dv['pv']) +
                               (Wdg_a[ts] + Wdg_b[ts])/(epsilon + cems.dv['dg'])) for ts in TS_range)

    # Equity Objective
    # Equity metrics calculations
    eta_ea = model.addVar(name='eta_ea', vtype=GRB.CONTINUOUS)
    eta_er = model.addVar(name='eta_er', vtype=GRB.CONTINUOUS)
    eta_es = model.addVar(name='eta_es', vtype=GRB.CONTINUOUS)
    model.addConstr(eta_ea == sum((1 + mgi[i].alpha) * sum(V[i, t] / (epsilon + Ymax[i][t]) for t in T_range) for i in I_range) / (2 * I * T))  # Energy Access
    model.addConstr(eta_er == 1 - sum((1 + mgi[i].alpha) * sum((Ymax[i][t] - V[i, t]) / Lmax[i][t] for t in T_range) for i in I_range) / (2 * I * T))  # Energy Resilience
    model.addConstr(eta_es == sum((1 + mgi[i].alpha) * sum(V[i, t] * (Xdg[i][t] / (epsilon + Z[i][t])) for t in T_range) for i in I_range)/
                    (2 * sum(epsilon + Ymax[i][t] for i in I_range for t in T_range)))  # Energy Resilience

    # Objective Function
    model.setObjective(eta_ea + eta_er + eta_es - Qc, sense=GRB.MAXIMIZE)

    return model


def buildRTCEMS(mgi, cems, PH):
    # Define Ranges
    T_range = range(PH)
    I_range = range(len(mgi.keys()))
    IT_range = [(i, t) for i in I_range for t in T_range]
    IJT_range = [(i, j, t) for i in I_range for j in I_range for t in T_range]

    model = gp.Model('RTCEMS')
    model.setParam('OutputFlag', 0)
    # Decision variables (stage 1)
    V = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='V')  # V: allocated power at time t to MG i
    a_V = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='a_V')  # a var for absolute values
    b_V = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='b_V')  # b var for absolute values

    Wes = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wes')  # Wes: power from ES shared resource at time t
    Wpv = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wpv')  # Wpv: power from PV shared resource at time t
    Wdg = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wdg')  # Wdg: power from DG shared resource at time t
    R = model.addVars(IT_range, vtype=GRB.CONTINUOUS, name='R')  # R: allocated power from shared resources at time t to MG i
    K = model.addVars(IJT_range, vtype=GRB.CONTINUOUS, name='K')  # K: power transitioned from MG i to j at time t
    E = model.addVars(range(PH+1), vtype=GRB.CONTINUOUS, name='E')
    Wpv_es = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wpv_es')  # Wpv_es: charging es with pv
    Wdg_es = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Wdg_es')  # Wpv_es: charging es with pv

    # Constraints (stage 1)
    # Update Needed. Maximum power allowed to be allocated to each community i. RHS: Ymax
    model.addConstrs((V[i, t] <= 0 for i in I_range for t in T_range), name='Vmax')
    # Update Needed. Z is provided at real time by other MGs. RHS: Z
    model.addConstrs((sum(K[i, j, t] for j in I_range) <= 0 for i in I_range for t in T_range), name='Koutput')
    # Update Needed. PV power components. RHS: PV
    model.addConstrs(((Wpv[t] + Wpv_es[t]) / cems.dv['pv'] <=  0 for t in T_range), name='PVPred')
    # Update Needed. The first energy level in ES at the begining of the prediction horizon. RHS: E0
    model.addConstr(E[0] == 0, name='E0')
    # Update Needed. RHS: V
    model.addConstrs((V[it] - (a_V[it] - b_V[it]) == 0 for it in IT_range), name='VPred')

    # Balance for shared resources
    model.addConstrs(sum(R[i, t] for i in I_range) == Wes[t] + Wpv[t] + Wdg[t] for t in T_range)
    # Lower & Upper limit on power amount to be assigned to MG
    model.addConstrs(mgi[i].Klimits['lower'] - V[i, t] <= 0 for i in I_range for t in T_range)
    model.addConstrs(V[i, t] <= mgi[i].Klimits['upper'] for i in I_range for t in T_range)
    # ES level feasibility.
    model.addConstrs(E[t + 1] == E[t] + cems.effic['es effic'] * (Wpv_es[t] + Wdg_es[t]) -
                     Wes[t] / cems.effic['es effic'] for t in T_range)
    # DG power components.
    model.addConstrs(Wdg[t] + Wdg_es[t] <= cems.effic['dg effic'] * cems.dv['dg'] for t in T_range)
    # V components.
    model.addConstrs(V[i, t] == R[i, t] + sum(K[j, i, t] for j in I_range) for i in I_range for t in T_range)
    # ES discharge limit
    model.addConstrs(Wes[t] <= cems.dv['es'] for t in T_range)

    # Objective function.
    model.setObjective(cems.gamma['adp'] * sum(a_V[it] + b_V[it] for it in IT_range), sense=GRB.MINIMIZE)

    model.update()
    return model


