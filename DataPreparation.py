"""
This python file builds the pkl file of each individual MG.
This is required as we want to save the input data of each microgrid separately.
"""
import pandas as pd
import numpy as np
import math


def getMGData(n_mgs, id, horizon_len):
    data = pd.read_csv('SystemInfo/data_microgrids.csv', index_col='mg number')
    private_data = {c: data[c].iloc[id] for c in data.columns}
    private_data['n_mgs'] = n_mgs
    private_data['id'] = id
    private_data['HL'] = horizon_len

    # scenario data
    private_data['probs'] = np.array(pd.read_csv(f'SystemInfo/Probs.csv')['prob'].values)
    private_data['n_scen'] = len(private_data['probs'])
    sd = pd.read_csv(f'SystemInfo/PV_{id}.csv')
    private_data['pv_s'] = np.array([np.array(sd[f'scen0'].values[:horizon_len])
                               for _ in range(private_data['n_scen'])])
    sd = pd.read_csv(f'SystemInfo/Load_{id}.csv')
    private_data['l_s'] = np.array([private_data['n_households'] *
                                    np.array(sd[f'scen{s}'].values[:horizon_len])
                                    for s in range(private_data['n_scen'])])
    # max load
    private_data['l_max'] = np.max(private_data['l_s'])
    private_data['l_mean'] = np.average(private_data['l_s'],
                                        weights=private_data['probs'], axis=0)[:horizon_len]

    # regulatory data
    rd = pd.read_csv(f'SystemInfo/data_regulatory.csv')
    private_data['u_cost'] = rd['infrastructure-usage(kw)'].iloc[0] + rd['metering(kw)'].iloc[0] + rd['fixed-fees(kw)'].iloc[0]
    private_data['dg_cost'] = rd['GHG-penalty(kw)'].iloc[0] + rd['fuel(kw)'].iloc[0]
    private_data['PNA_cost'] = rd['PNA_cost'].iloc[0]
    private_data['CO2'] = rd['co2(kg/kwh)'].iloc[0]


    private_data['C_t_min'] = -horizon_len * private_data['pay_max'] * (private_data['es'] + private_data['pv'] + private_data['dg'])
    private_data['C_t_max'] = horizon_len * \
                         (private_data['pay_max'] * (private_data['es'] + private_data['l_max']) +
                          private_data['dg_cost'] * private_data['dg'] +
                          private_data['es_cost'] * private_data['es'] +
                          private_data['lsp'] * private_data['l_max'])


    subsidy_data = pd.read_csv(f'SystemInfo/data_regulatory.csv')
    private_data['fsrr'] = subsidy_data['financial subsidy rate'].iloc[0] * private_data['sv']
    private_data['usrr'] = subsidy_data['utility subsidy rate'].iloc[0] * private_data['sv']

    # public data
    public_data = {'fsrr':private_data['fsrr'], 'usrr': private_data['usrr']}

    return private_data, public_data


def getCEMSData(num_mgs, mg_public, horizon_len):
    d = {}
    d['n_mgs'] = num_mgs
    d['HL'] = horizon_len
    d['fsrr'] = [mg_public[mg_id]['fsrr'] for mg_id in mg_public.keys()]
    d['usrr'] = [mg_public[mg_id]['usrr'] for mg_id in mg_public.keys()]
    rd = pd.read_csv(f'SystemInfo/data_regulatory.csv')
    d['u_cost'] = rd['infrastructure-usage(kw)'].iloc[0] + rd['metering(kw)'].iloc[0] + rd['fixed-fees(kw)'].iloc[0]
    d['TFS'] = rd['total financial support'].iloc[0]
    d['TUS'] = rd['total utility support'].iloc[0]

    return d
