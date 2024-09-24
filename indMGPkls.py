"""
This python file builds the pkl file of each individual MG.
This is required as we want to save the input data of each microgrid separately.
"""

import pickle as pkl

import pandas as pd

if __name__ == "__main__":
    # number of microgrids is saved as MG
    MG = 3

    # gamma: gamma is a dictionary of all penalties that are necessary for the objective function calculation.
    # The key is microgrid number, the values (v) are penalties in order: gamma_Bid, gamma_LSP, gamma_SDP which are
    # Bidding, Load Shedding, and Scheduling Deviation.
    v = [[0.7, 1.2, 0.7], [0.7, 0.8, 0.8], [0.7, .2, 0.5]]
    gamma = {i: {'bid': v[i][0], 'lsp': v[i][1], 'sdp': v[i][2]} for i in range(MG)}

    # dv: device specifications are stored in this variables in a dictionary format.
    # The key is microgrid number, the values (v) are device capacities in order: ES, PV, DG.
    v = [[500, 800, 20], [400, 700, 40], [1200, 1200, 60]]
    dv = {i: {'es': v[i][0], 'pv': v[i][1], 'dg': v[i][2]} for i in range(MG)}

    # ps: probability of scenarios is saved in this variable to turn into a pickle file.
    # The key is the scenario number, and the values are the probabilities in array format.
    # Note that each microgrid has its own probabilities.
    ps = {i: pd.read_csv(f'MGs/ps{i}.csv')['prob'].values for i in range(MG)}

    # pv_s: this is a dictionary containing the scenarios for pv. The number of scenarios must be aligned with ps.
    # The key is scenario number, the values are dictionaries again with scenarios in array format for each scenario.
    # Note that the generated scenarios are for 1kW PV device, hence the values must be multiplied by PV capacity.
    pv_s = {}
    for i in range(MG):
        v = pd.read_csv(f'MGs/pv_s{i}.csv') * dv[i]['pv']
        pv_s[i] = {j: v[f'scen{j}'].values for j in range(len(ps[i]))}

    # l_s: this is a dictionary containing the scenarios for load. The number of scenarios must be aligned with ps.
    # The key is scenario number, the values are dictionaries again with scenarios in array format for each scenario.
    # l_max: maximum predicted load demand value at each time step.
    l_s, l_max = {}, {}
    for i in range(MG):
        v = pd.read_csv(f'MGs/l_s{i}.csv')
        l_s[i] = {j: v[f'scen{j}'].values for j in range(len(ps[i]))}
        l_max[i] = {t: max(v.iloc[t]) for t in range(len(v))}


    # effi: it is a dictionary containing the efficiency of devices.
    # The order is ES charge/discahrge efficiency, PV efficiency, and DG efficiency.
    v = [[0.8, 0.4, 0.8], [0.9, 0.8, 0.8], [0.95, 0.3, 0.8]]
    effi = {i: {'es': v[i][0], 'pv': v[i][1], 'dg': v[i][2]} for i in range(MG)}

    for i in range(MG):
        with open(f'pkls/mg{i}.pkl', 'wb') as handle:
            pkl.dump([gamma[i], dv[i], ps[i], pv_s[i], l_s[i], l_max[i], effi[i]], handle)
        handle.close()