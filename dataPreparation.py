"""
This python file builds the pkl file of each individual MG.
This is required as we want to save the input data of each microgrid separately.
"""
import pandas as pd
import numpy as np


def getMGData(i):
    data = pd.read_csv('systemInfo/data.csv', index_col='mg number')
    mg_dict = {}
    mg_dict['es'] = data['es'].iloc[i]
    mg_dict['pv'] = data['pv'].iloc[i]
    mg_dict['dg'] = data['dg'].iloc[i]
    mg_dict['es_c'] = data['es charge effic'].iloc[i]
    mg_dict['es_d'] = data['es discharge effic'].iloc[i]
    mg_dict['dg_ef'] = data['dg effic'].iloc[i]
    mg_dict['lsp'] = data['lsp'].iloc[i]
    mg_dict['es_cost'] = data['es cost'].iloc[i]
    mg_dict['doc'] = data['doc'].iloc[i]
    mg_dict['dod'] = data['dod'].iloc[i]
    mg_dict['sv'] = data['sv'].iloc[i]
    mg_dict['alpha'] = data['alpha'].iloc[i]
    mg_dict['pay_min'] = data['pay min'].iloc[i]
    mg_dict['pay_max'] = data['pay max'].iloc[i]

    # scenario data
    mg_dict['probs'] = np.array(pd.read_csv(f'systemInfo/probabilities_{i}.csv')['prob'].values)
    sd = pd.read_csv(f'systemInfo/pv_scenarios_{i}.csv')
    mg_dict['pv_s'] = np.array([np.array(sd[f'scen{s}'].values) for s in range(len(mg_dict['probs']))])
    sd = pd.read_csv(f'systemInfo/load_scenarios_{i}.csv')
    mg_dict['l_s'] = np.array([np.array(sd[f'scen{s}'].values) for s in range(len(mg_dict['probs']))])
    mg_dict['l_max'] = max(max(mg_dict['l_s']))

    # regulatory data
    rd = pd.read_csv(f'systemInfo/regulatory data.csv')
    mg_dict['u_fee'] = rd['infrastructure-usage(kw)'].iloc[0] + rd['metering(kw)'].iloc[0] + rd['fixed-fees(kw)'].iloc[0]
    mg_dict['dg_cost'] = rd['GHG-penalty(kw)'].iloc[0] + rd['fuel(kw)'].iloc[0]
    mg_dict['PNA_cost'] = rd['PNA_cost'].iloc[0]

    # disagreement points
    mg_dict['eta_r_Non'] = 1
    mg_dict['eta_c_Non'] = 1
    return mg_dict


def getNetwork():
    nd = pd.read_csv(f'systemInfo/network power limits.csv')
    min_trans, max_trans = {}, {}
    for k in range(len(nd)):
        min_trans[(nd['source'].iloc[k], nd['sink'].iloc[k])] = nd['min limit'].iloc[k]
        max_trans[(nd['source'].iloc[k], nd['sink'].iloc[k])] = nd['max limit'].iloc[k]
    return min_trans, max_trans
