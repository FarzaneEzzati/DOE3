import gurobipy as gp
from gurobipy import GRB


def buildDAMG(mg, SH):  # buildForModel builds the forecast model of individual microgrids

    # Let's set the time periods and scenarios automatically based on the provided inputs
    T = SH
    T_range = range(T)  # range of time periods e.g. [1, 2, 3, 4, 5, 6]
    T1_range = range(T+1)
    S = len(mg.ps)  # number of scenarios e.g. 4
    S_range = range(S)  # range of scenarios e.g. [1, 2, 3, 4]
    TS_range = [(t, s) for s in S_range for t in T_range]  # the list of indices for decisions variables in stage 2
    T1S_range = [(t, s) for s in S_range for t in T1_range]  # For ES

    model = gp.Model('Forecast')
    model.setParam('OutPutFlag', 0)
    # Decision variables (stage 1)
    Z = model.addVars(T_range,  vtype=GRB.CONTINUOUS, name='Z')  # Bidding decision
    Xes = model.addVars(T_range,  vtype=GRB.CONTINUOUS, name='Xes')  # Bidding from ES
    Xpv = model.addVars(T_range,  vtype=GRB.CONTINUOUS, name='Xpv')  # Bidding from PV
    Xdg = model.addVars(T_range,  vtype=GRB.CONTINUOUS, name='Xdg')  # Bidding from DG

    # Decision variables (stage 2)
    E = model.addVars(T1S_range, vtype=GRB.CONTINUOUS, name='E')
    Z_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Z_')  # Bidding decision
    Xpv_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xpv_es')  # power from pv to es to store
    Xpv_l = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xpv_l')  # power from pv to load
    Xpv_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xpv_')  # power from pv to exchange under scen s

    Xdg_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xdg_es')  # power from dg to es to store
    Xdg_l = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xdg_l')  # power from dg to load
    Xdg_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xdg_')  # power from dg to exchange under scen s

    Xes_l = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xes_l')  # power from dg to es to store
    Xes_ = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='Xes_')  # power from es to exchanges under scen s

    Y = model.addVars(TS_range,  vtype=GRB.CONTINUOUS, name='Y')  # load to curtail
    # Decision variables (model absolute values of Delta)
    a_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='a_es')  # a to be in a+b as objective
    b_es = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='b_es')  # b to be in a+b as objective
    a_pv = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='a_pv')  # a to be in a+b as objective
    b_pv = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='b_pv')  # b to be in a+b as objective
    a_dg = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='a_dg')  # a to be in a+b as objective
    b_dg = model.addVars(TS_range, vtype=GRB.CONTINUOUS, name='b_dg')  # b to be in a+b as objective

    # Constraints for stage 1
    model.addConstrs(Z[t] == Xes[t] + Xpv[t] + Xdg[t] for t in T_range)
    model.addConstrs(Xes[t] <= mg.dv['es'] for t in T_range)
    model.addConstrs(Xpv[t] <= mg.dv['pv'] for t in T_range)
    model.addConstrs(Xdg[t] <= mg.dv['dg'] for t in T_range)

    # Constraints for stage 2
    # ES level feasibility constraints
    model.addConstrs(E[0, s] == mg.E0 * mg.dv['es'] for s in S_range)
    model.addConstrs(E[t + 1, s] == E[t, s] + mg.effic['es effic'] * (Xpv_es[t, s] + Xdg_es[t, s]) -
                     (Xes_l[t, s] + Xes_[t, s]) / mg.effic['es effic'] for s in S_range for t in T_range)
    # ES level should follow ES capacity
    model.addConstrs(E[ts] <= mg.dv['es'] for ts in TS_range)
    # Load components
    model.addConstrs(Xes_l[ts] + Xpv_l[ts] + Xdg_l[ts] + Y[ts] == mg.l_s[ts[1]][ts[0]]
                     for ts in TS_range)
    # PV generated power components
    model.addConstrs(Xpv_l[ts] + Xpv_es[ts] + Xpv_[ts] <= mg.dv['pv'] * mg.pv_s[ts[1]][ts[0]]
                     for ts in TS_range)

    # DG generated power components
    model.addConstrs(Xdg_l[ts] + Xdg_es[ts] + Xdg_[ts] <= mg.effic['dg effic'] * mg.dv['dg']
                     for ts in TS_range)

    # ES level limit
    model.addConstrs(Xes_l[ts] + Xes_[ts] <= 0.9 * mg.dv['es'] for ts in TS_range)
    model.addConstrs(Xpv_es[ts] + Xdg_es[ts] <= 0.9 * mg.dv['es'] for ts in TS_range)

    # Deviation calculations
    model.addConstrs(Xes[ts[0]] - Xes_[ts] == a_es[ts] - b_es[ts] for ts in TS_range)
    model.addConstrs(Xpv[ts[0]] - Xpv_[ts] == a_pv[ts] - b_pv[ts] for ts in TS_range)
    model.addConstrs(Xdg[ts[0]] - Xdg_[ts] == a_dg[ts] - b_dg[ts] for ts in TS_range)

    # Set objective function which is a monetary amount from bidding revenue and load shedding
    total_objective = mg.gamma['bid'] * sum(Z[t] for t in T_range) - \
                      mg.gamma['lsp'] * sum(mg.ps[ts[1]] * Y[ts] for ts in TS_range) - \
                      mg.gamma['sdp'] * sum(mg.ps[ts[1]] * ((a_es[ts] + b_es[ts]) + (a_pv[ts] + b_pv[ts]) + (a_dg[ts] + b_dg[ts])) for ts in TS_range)
    model.setObjective(total_objective, sense=GRB.MAXIMIZE)
    return model


def buildRTMG(mg, PH):
    T_range = range(PH)
    model = gp.Model('RealTime')
    model.setParam('OutPutFlag', 0)

    # Decision variables
    a_Z = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='a_Z')  # These are variables to linearize absolute value.
    b_Z = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='b_Z')

    Z = model.addVars(T_range, ub=mg.dv['es'], vtype=GRB.CONTINUOUS, name='Z')  # Bidding decision in real time

    Ves = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Ves')  # Power provided by central EMS to store in es
    Vl = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Vl')  # Power provided by central EMS to serve load
    E = model.addVars(range(PH+1), vtype=GRB.CONTINUOUS, name='E')
    Y = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Y')  # Load shed
    Xes = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xes')  # Bidding from ES
    Xpv = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xpv')  # Bidding from PV
    Xdg = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xdg')  # Bidding from DG
    Xpv_es = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xpv_es')  # power from pv to es to store
    Xpv_l = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xpv_l')  # power from pv to load
    Xdg_es = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xdg_es')  # power from dg to es to store
    Xdg_l = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xdg_l')  # power from dg to load
    Xes_l = model.addVars(T_range, vtype=GRB.CONTINUOUS, name='Xes_l')  # power from dg to es to store

    # Constraints for real time operation
    # Update Needed. This constraint's rhs is an input to the model. Initialized with 0, then changes.
    model.addConstr(E[0] == 0, name='E0')
    # ES level feasibility.
    model.addConstrs(E[t+1] == E[t] + mg.effic['es effic'] * (Xpv_es[t] + Xdg_es[t] + Ves[t]) -
                                (Xes_l[t] + Xes[t]) / mg.effic['es effic'] for t in T_range)
    # Update Needed. Load components in real time. Load value is 0 but changes in the sequential operation.
    model.addConstrs((Xes_l[t] + Xpv_l[t] + Xdg_l[t] + Vl[t] + Y[t] <= 0 for t in T_range), name='LoadPred')
    # Update Needed. PV generated power components. This constraint's rhs is an input to the model. Initialized with 0, then changes.
    model.addConstrs(((Xpv_es[t] + Xpv_l[t] + Xpv[t])/mg.dv['pv'] <= 0 for t in T_range), name='PVPred')
    # DG generated power components. This constraint's rhs is fixed.
    model.addConstrs((Xdg_es[t] + Xdg_l[t] + Xdg[t] <= mg.dv['dg'] * mg.effic['dg effic'] for t in T_range), name='DG power')
    # Z in real time changes.
    model.addConstrs((Z[t] == Xes[t] + Xpv[t] + Xdg[t] for t in T_range), name='ZReal')
    # Update Needed. Absolute value of Z - Zf. Zf must be updated in each iteration.
    model.addConstrs((Z[t] - (a_Z[t] - b_Z[t]) == 0 for t in T_range), name='ZPred')
    # Update Needed. Components of V assigned to the MG. Vf is the forecasted value.
    model.addConstrs((Ves[t] + Vl[t] == 0 for t in T_range), name='VElements')
    # Objective function which is the absolute value of Z - Z_f
    model.setObjective(mg.gamma['sdp'] * sum(a_Z[t] + b_Z[t] for t in T_range), sense=GRB.MINIMIZE)
    model.update()
    return model
