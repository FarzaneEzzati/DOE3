"""
This python file builds the pkl file of CEMS.
This is required as we want to save the input data of CEMS separately.
The data include shared resources capacities, scenarios for PV, device efficiencies, penalties.
"""

import pickle as pkl

import pandas as pd

if __name__ == "__main__":
    # gamma: gamma is a dictionary of all penalties that are necessary for the objective function calculation.
    # The key is penalty name, the values (v) are penalties. We only have adp (allocation deviation penalty) for now.
    gamma = {'adp': 1.3}

    # dv: device specifications are stored in this variables in a dictionary format.
    # The key is device name, the values (v) are device capacities in order: ES, PV, DG.
    dv = {'es': 2800, 'pv': 3000, 'dg': 80}

    # ps: probability of scenarios is saved in this array variable to turn into a pickle file.
    ps = pd.read_csv(f'MGs/ps{1}.csv')['prob'].values

    # pv_s: this is a dictionary containing the scenarios for pv. The number of scenarios must be aligned with ps.
    # The key is scenario number, the values are dictionaries again with scenarios in array format for each scenario.
    # Note that the generated scenarios are for 1kW PV device, hence the values must be multiplied by PV capacity.
    v = pd.read_csv(f'MGs/pv_s{1}.csv') * dv['pv']
    pv_s = {j: v[f'scen{j}'].values for j in range(len(ps))}

    # effi: it is a dictionary containing the efficiency of devices.
    # The order is ES charge/discahrge efficiency, PV efficiency, and DG efficiency.
    effi = {'es': 0.8, 'pv': 0.5, 'dg': 0.8}

    with open(f'pkls/cems.pkl', 'wb') as handle:
        pkl.dump([gamma, dv, ps, pv_s, effi], handle)
    handle.close()