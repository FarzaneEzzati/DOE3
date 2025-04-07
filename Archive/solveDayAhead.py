'''This file contains a function solving day ahead models and returns optimized decisions'''
from buildCEMS import buildDACEMS  # import functions to build models for individual MGs
from buildMG import buildDAMG  # import functions to build models for CEMS
import pickle as pkl



def solveDAMGs(mgi, SH):
    # We build the day-ahead models for each microgrid separately (named mgm: microgrid models).
    # Save the optimal solution as well in a dictionary (named mgs: microgrid solutions).
    T_range = range(SH)
    mgm = {}
    mg_Z, mg_Lmax, mg_Ymax = {}, {}, {}
    mg_Xes, mg_Xpv, mg_Xdg = {}, {}, {}
    for mg in mgi.keys():
        mgm[mg] = buildDAMG(mgi[mg], SH)
        mgm[mg].optimize()

        # Get Z and Z_dg from obtained solutions.
        mg_Z[mg] = {t: mgm[mg].getVarByName(f'Z[{t}]').x for t in T_range}
        mg_Xes[mg] = {t: mgm[mg].getVarByName(f'Xes[{t}]').x for t in T_range}
        mg_Xpv[mg] = {t: mgm[mg].getVarByName(f'Xpv[{t}]').x for t in T_range}
        mg_Xdg[mg] = {t: mgm[mg].getVarByName(f'Xdg[{t}]').x for t in T_range}

        # Get maximum load shedding from solutions and load amount from information available.
        mg_Lmax[mg] = mgi[mg].l_max
        mg_Ymax[mg] = {t: max([mgm[mg].getVarByName(f'Y[{t},{s}]').x for s in range(len(mgi[mg].ps))]) for t in T_range}

    # Save results for the day
    with open('../Performance/dayAheadMGs.pkl', 'wb') as handle:
        pkl.dump([mg_Ymax, mg_Lmax, mg_Z, mg_Xes, mg_Xpv, mg_Xdg], handle)
    handle.close()


def solveDACEMS(cems, mgi, SH):
    I = len(mgi.keys())
    I_range = range(I)
    T_range = range(SH)
    # Load results for the day
    with open('../Performance/dayAheadMGs.pkl', 'rb') as handle:
        mg_Ymax, mg_Lmax, mg_Z, mg_Xes, mg_Xpv, mg_Xdg = pkl.load(handle)
    handle.close()

    # Upload information for the CEMS.
    cems_model = buildDACEMS(cems, mg_Z, mg_Xdg, mg_Lmax, mg_Ymax, mgi, SH)
    cems_model.optimize()

    # Save results to be plotted as Day-Ahead Planning.
    _EA = cems_model.getVarByName('eta_ea').x  # Energy Access
    _ER = cems_model.getVarByName('eta_er').x  # Energy Resilience
    _ES = cems_model.getVarByName('eta_es').x  # Energy Sustainability
    K = {(i, j): {} for i in I_range for j in I_range}
    R = {i: {} for i in I_range}
    V = {i: {} for i in I_range}
    for i in I_range:
        for t in T_range:
            for j in I_range:
                K[(i, j)][t] = cems_model.getVarByName(f'K[{i},{j},{t}]').x
            V[i][t] = cems_model.getVarByName(f'V[{i},{t}]').x
            R[i][t] = cems_model.getVarByName(f'R[{i},{t}]').x

    with open('../Performance/dayAheadCEMS.pkl', 'wb') as handle:
        pkl.dump([V, R, K, _EA, _ER, _ES], handle)
    handle.close()


