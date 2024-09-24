import gurobipy as gp
from gurobipy import GRB



def buildForMG(mg_input):  # buildForModel builds the forecast model of individual microgrids

    # Let's set the time periods and scenarios automatically based on the provided inputs
    T = len(mg_input.l_s[1])  # number of time periods e.g. 6 hours
    T_range = range(T)  # range of time periods e.g. [1, 2, 3, 4, 5, 6]
    S = len(mg_input.ps)  # number of scenarios e.g. 4
    S_range = range(S)  # range of scenarios e.g. [1, 2, 3, 4]
    TS_range = [(t, s) for s in S_range for t in T_range]  # the list of indices for decisions variables in stage 2

    model = gp.Model('Forecast')
    model.setParam('OutPutFlag', 0)
    # Decision variables (stage 1)
    Z = model.addVars(T_range, lb=0, vtype=GRB.CONTINUOUS, name='Z')  # Bidding decision
    Zes = model.addVars(T_range, lb=0, vtype=GRB.CONTINUOUS, name='Zes')  # Bidding from ES
    Zpv = model.addVars(T_range, lb=0, vtype=GRB.CONTINUOUS, name='Zpv')  # Bidding from PV
    Zdg = model.addVars(T_range, lb=0, vtype=GRB.CONTINUOUS, name='Zdg')  # Bidding from DG

    # Decision variables (stage 2)
    E = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='E')
    Xpv_es = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xpv_es')  # power from pv to es to store
    Xpv_l = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xpv_l')  # power from pv to load
    Xpv_c = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xpv_c')  # power from pv to curtail

    Xdg_es = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xdg_es')  # power from dg to es to store
    Xdg_l = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xdg_l')  # power from dg to load
    Xdg_c = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xdg_c')  # power from dg to curtail

    Xes_l = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Xes_l')  # power from dg to es to store

    Y = model.addVars(TS_range, lb=0, vtype=GRB.CONTINUOUS, name='Y')  # load to curtail

    # Constraints for stage 1
    model.addConstrs(Z[t] == Zes[t] + Zpv[t] + Zdg[t] for t in T_range)
    model.addConstrs(Zes[t] <= mg_input.dv['es'] for t in T_range)
    model.addConstrs(Zpv[t] <= mg_input.dv['pv'] for t in T_range)
    model.addConstrs(Zdg[t] <= mg_input.dv['dg'] for t in T_range)

    # Constraints for stage 2
    # ES level feasibility constraints
    model.addConstrs(E[t + 1, s] == E[t, s] + mg_input.effi['es']*(Xpv_es[t, s] + Xdg_es[t, s]) -
                                    (Xes_l[t, s] + Zes[t])/mg_input.effi['es'] for s in S_range for t in T_range[:-1])
    # ES level should follow ES capacity
    model.addConstrs(E[ts] <= mg_input.dv['es'] for ts in TS_range)
    # Load components
    model.addConstrs(Xes_l[t, s] + Xpv_l[t, s] + Xdg_l[t, s] + Y[t, s] == mg_input.l_s[s][t]
                     for s in S_range for t in T_range)
    # PV generated power components
    model.addConstrs(Xpv_l[t, s] + Xpv_es[t, s] + Xpv_c[t, s] + Zpv[t] == mg_input.pv_s[s][t]
                     for s in S_range for t in T_range)
    # DG generated power components
    model.addConstrs(Xdg_l[t, s] + Xdg_es[t, s] + Xdg_c[t, s] + Zdg[t] == mg_input.effi['dg'] * mg_input.dv['dg']
                     for s in S_range for t in T_range)

    # Set objective function which is a monetary amount from bidding revenue and load shedding
    total_objective = mg_input.gamma['bid'] * sum(Z[t] for t in T_range) - \
          sum(mg_input.ps[s] * mg_input.gamma['lsp'] * sum(Y[t, s] for t in T_range) for s in S_range)
    model.setObjective(total_objective, sense=GRB.MAXIMIZE)
    return model


def buildRealMG(mg_input):

    model = gp.Model('RealTime')
    model.setParam('OutPutFlag', 0)
    # Decision variables (stage 1)
    A = model.addVars((1, 2), lb=0, vtype=GRB.CONTINUOUS, name='A')  # These are variables to linearize absolute value.
    Z = model.addVar(lb=0, ub=mg_input.dv['es'], vtype=GRB.CONTINUOUS, name='Z')  # Bidding decision in real time
    Zes = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Zes')  # Bidding from ES
    Zpv = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Zpv')  # Bidding from PV
    Zdg = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Zdg')  # Bidding from DG

    Ves = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Ves')  # Power provided by central EMS to store in es
    Vl = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Vl')  # Power provided by central EMS to serve load

    # Decision variables (stage 2)
    E = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='E')
    Xpv_es = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Xpv_es')  # power from pv to es to store
    Xpv_l = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Xpv_l')  # power from pv to load

    Xdg_es = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Xdg_es')  # power from dg to es to store
    Xdg_l = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Xdg_l')  # power from dg to load

    Xes_l = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='Xes_l')  # power from dg to es to store

    # Constraints for real time operation
    # ES level feasibility. This constraint's rhs is an input to the model. Initialized with 0, then changes.
    model.addConstr(E - (mg_input.effi['es']*(Xpv_es + Xdg_es + Ves) - (Xes_l + Zes)/mg_input.effi['es']) == 0,
                    name='E level')
    # Load components in real time. Load value is 0 but changes in the sequential operation.
    model.addConstr(Xes_l + Xpv_l + Xdg_l + Vl <= 0, name='Load components')
    # PV generated power components. This constraint's rhs is an input to the model. Initialized with 0, then changes.
    model.addConstr(Xpv_es + Xpv_l + Zpv <= 0, name='PV power')
    # DG generated power components. This constraint's rhs is fixed.
    model.addConstr(Xdg_es + Xdg_l + Zdg <= mg_input.dv['dg']*mg_input.effi['dg'], name='DG power')
    # Z in real time changes.
    model.addConstr(Z == Zes + Zpv + Zdg, name='Z components')
    # Absolute value of Z - Zf. Zf must be updated in each iteration.
    model.addConstr(Z - (A[1] - A[2]) == 0, name='abs')
    # Components of V assigned to the MG. Vf is the forecasted value.
    model.addConstr(Ves + Vl == 0, name='V components')

    # Objective function which is the absolute value of Z - Z_f
    model.setObjective(mg_input.gamma['sdp'] * (A[1] + A[2]), sense=GRB.MINIMIZE)
    model.update()
    return model