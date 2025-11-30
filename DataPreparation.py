"""
This python file builds the pkl file of each individual MG.
This is required as we want to save the input data of each microgrid separately.
"""
import pandas as pd
import numpy as np
import math


def getMGData(n):
    data = pd.read_csv('SystemInfo\data_microgrids.csv', index_col='mg number')
    private_data = {c: data[c].iloc[n] for c in data.columns}
    private_data['n'] = n
    private_data['alpha'] = 0
    # scenario data
    sd = pd.read_csv(f'SystemInfo/PV.csv')
    private_data['pv_hourly'] = sd[f'PV_{n}'].values
    sd = pd.read_csv(f'SystemInfo/Load.csv')
    private_data['load'] = sd[f'load_{n}'].values * private_data['n_households']
    private_data['load_price'] = (1 + private_data['wtp']) * private_data['grid_price']

    # regulatory data
    rd = pd.read_csv(f'SystemInfo/data_regulatory.csv')
    private_data['u_cost'] = rd['infrastructure-usage(kw)'].iloc[0] + rd['metering(kw)'].iloc[0] + rd['fixed-fees(kw)'].iloc[0]
    private_data['dg_cost'] = rd['GHG-penalty(kw)'].iloc[0] + rd['fuel(kw)'].iloc[0]
    private_data['PNA_cost'] = rd['PNA_cost'].iloc[0]
    private_data['CO2'] = rd['co2(kg/kwh)'].iloc[0]

    private_data['C_t_min'] = -2000
    private_data['C_t_max'] = 2000

    private_data['r_priority'] = 0.75
    private_data['c_priority'] = 0.25

    rd = pd.read_csv(f'SystemInfo/data_regulatory.csv')
    private_data['fsrr'] = rd['financial subsidy rate'].iloc[0] * (0.5 + private_data['sv']) / 1.5
    private_data['usrr'] = rd['utility subsidy rate'].iloc[0] * (0.5 + private_data['sv']) / 1.5    
    private_data['TFS'] = rd['total financial support'].iloc[0]
    private_data['TUS'] = rd['total utility support'].iloc[0]
    private_data['u_cost'] = rd['infrastructure-usage(kw)'].iloc[0] + rd['metering(kw)'].iloc[0] + rd['fixed-fees(kw)'].iloc[0]
    
    # Public data
    public_data = {'fsrr':private_data['fsrr'], 
                   'usrr': private_data['usrr'], 
                   'TFS': private_data['TFS'],
                   'TUS': private_data['TUS'],
                   'u_cost': private_data['u_cost']}

    return private_data, public_data


def buildCEMSData(public_data):
    cems_data = {}
    cems_data['fsrr'] = np.array([mg['fsrr'] for mg in public_data])
    cems_data['usrr'] = np.array([mg['usrr'] for mg in public_data])
    cems_data['TFS'] = public_data[0]['TFS']
    cems_data['TUS'] = public_data[0]['TUS']
    cems_data['u_cost'] = public_data[0]['u_cost']
    return cems_data
