'''
This file contains functions to facilitate getting desired tables/figures from results.
'''
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict


class MGResults:
    def __init__(self, N, T, n, data_private):
        with open(f'Solution/(N={N},T={T},n={n})solu.pkl', 'rb') as handle:
            self.solu = pickle.load(handle)
        self.T = T
        self.data_private = data_private
    
    def get_opr_cost(self):
        return self.solu['C_es'] + self.solu['C_sh'] + self.solu['C_dg']
    
    def get_payments(self):
        return (self.solu['pi_b']- self.solu['pi_s']).sum()
    
    def get_kpi(self):
        # Connection rate
        did_buy = self.solu['e_b'] == 0
        did_shed = self.solu['e_b'] >= 0.5 * self.solu['e_b']
        CR = 100 * (1 - np.sum(did_shed) / self.T)

        # Renewable Energy Access Rate
        generated_pv = self.solu['g_pv'].sum()
        demanded_load = self.solu['l_m'].sum()
        REAR = 100 *  generated_pv/demanded_load

        # CO2 emission, 0.4–0.5 kg CO₂/kWh is grid baseline
        generated_dg = self.solu['g_dg'].sum()
        served_load = self.solu['l_m'].sum()
        dg_CO2 =  generated_dg * self.data_private['CO2']
        grid_CO2 = served_load * 0.45
        CO2 = 100 * (grid_CO2 - dg_CO2)/grid_CO2

        # Renewable Energy Mandate
        REM = 100 * generated_pv / (generated_pv + generated_dg)

        return CR, REAR, CO2, REM
    
    def get_utility_cost(self):
        return (self.solu['e_b'] + self.solu['e_s']).sum() * self.data_private['u_cost']
    
    def get_utility_subsidy_usage(self):
        utility_cost = self.get_utility_cost()
        return utility_cost * self.data_private['usrr']
    
    def get_purchase_subsidy_usage(self):
        buy_payment = np.sum(self.solu['pi_b'])
        return buy_payment * self.data_private['fsrr']
    
    def get_total_generation(self):
        return (self.solu['g_pv'] + self.solu['g_dg']).sum()
       
    def get_total_buy(self):
        return self.solu['e_b'].sum()
    
    def get_total_sell(self):
        return self.solu['e_s'].sum()
    
    def get_buy(self):
        return self.solu['pi_b']
    
    def get_sell(self):
        return self.solu['pi_s']


def plot_trade(results, N, T):
    for key, result in results.items():
        y = (result.solu['e_b'] - result.solu['e_s']).sum()
        y = np.append(y, y[-1])
        x = np.arange(y.shape[0])

        plt.figure(figsize=(5, 4))
        plt.step(x=x, y=y, where='post')
        plt.axhline(0, color='red', linestyle=':')
        plt.ylabel('Energy Traded (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.savefig(f'Figures/(N={N},T={T})trade({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_load_served(results, N, T):
    for key, result in results.items():
        y1 = result.solu['l_m']
        y1 = np.hstack((y1, y1[-1]))
        x = np.arange(y1.shape[0])
        
        y2 = result.solu['l_sh']
        y2 = np.hstack((y2, y2[-1]))

        plt.figure(figsize=(5, 2))
        plt.step(x=x, y=y1, where='post', label=f'Served Load', color='green')
        plt.step(x=x, y=y1+y2, where='post', label=f'Total Load', color='red', linestyle=':')
        plt.ylabel('Load (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.savefig(f'Figures/(N={N},T={T})load({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_trade(results, N, T):
    buy = np.stack([result.solu['e_b'] for result in results.values()])
    sell = np.stack([result.solu['e_s'] for result in results.values()])
    
    buy_sell = buy - sell
    buy_sell = np.sum(buy_sell, axis=1)
    buy_sell = np.concatenate((buy_sell, buy_sell[:, -1][:, np.newaxis]), axis=1)
    
    y_min, y_max = np.min(buy_sell), np.max(buy_sell)
    c = ['orange', 'red', 'blue']
    plt.rcParams['font.size'] = 12
    for n in results.keys():
        x, y = range(25), buy_sell[n]
        fig = plt.figure(figsize=(5, 2))
        plt.step(x, y, where='post', color=c[n], label=f'MG {n}')
        plt.fill_between(x, y, 0, step='post', color=c[n], alpha=0.3)
        plt.xticks(np.linspace(0, 24, 13))
        plt.plot([0, 24], [0, 0], ':', color='black')
        plt.ylabel('(kW/h)')
        plt.xlabel('Time')
        plt.ylim([1.05*y_min, 1.05*y_max])
        plt.title(f'MG {n+1}')
        plt.savefig(f'Figures/(N={N},T={T})trade(MG{n}).jpg', bbox_inches='tight', dpi=600)
        plt.close()


def get_cost_table(results, N, T):
    indices = [f'Community {n}' for n in range(N)] + ['System']
    table = {key: np.zeros(N) for key in ['Cost Before Trade ($)',
                                        'Cost After Trade ($)',
                                        'Utility Fee ($)',
                                        'Sell Paid ($)',
                                        'Buy Paid ($)',
                                        'Utility Subsidy Usage ($)',
                                        'Purchase Subsidy Usage ($)']}
    for n, result in results.items():
        table['Cost Before Trade ($)'][n] = result.solu['C_t_Non']
        table['Cost After Trade ($)'][n] = result.solu['C_t']
        table['Utility Fee ($)'][n] = result.get_utility_cost()
        table['Sell Paid ($)'][n] = result.solu['pi_s'].sum()
        table['Buy Paid ($)'][n] = result.solu['pi_b'].sum()
        table['Utility Subsidy Usage ($)'][n] = result.get_utility_subsidy_usage()
        table['Purchase Subsidy Usage ($)'][n] = result.get_purchase_subsidy_usage()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/(N={N},T={T})cost_table.csv')


def get_metric_table(results, N, T, alphas):
    indices = [f'Community {n}' for n in range(N)] + ['System']
    table = {key: np.zeros(N) for key in ['Resilience Before Trade',
                                        'Cost Before Trade',
                                        'Resilience After Trade',
                                        'Cost After Trade']}
    for n, result in results.items():
        table['Resilience Before Trade'][n] = result.solu['eta_r_Non']
        table['Resilience After Trade'][n] = result.solu['eta_r']
        table['Cost Before Trade'][n] = result.solu['eta_c_Non']
        table['Cost After Trade'][n] = result.solu['eta_c']
    # adding system wide
    table = {key: np.append(val, np.average(val, weights=alphas)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/(N={N},T={T})metric_table.csv')


def get_kpi_table(results, N, T):
    indices = [f'Community {n}' for n in range(N)] + ['System']
    table = {key: np.zeros(N) for key in ['Connection Rate', 'Renewable Energy Access Rate',
                                        'CO2 Emission Reduction', 'Renewable Energy Mandate']}
    for n, result in results.items():
        CR, REAR, CO2, REM = result.get_kpi()
        table['Connection Rate'][n] = CR
        table['Renewable Energy Access Rate'][n] = REAR
        table['CO2 Emission Reduction'][n] = CO2
        table['Renewable Energy Mandate'][n] = REM
    table = {key: np.append(val, np.average(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/(N={N},T={T})kpi_table.csv')


def get_power_table(results, N, T, alphas):
    indices = [f'Community {n}' for n in range(N)] + ['System']
    table = {key: np.zeros(N) for key in ['Generation', 'Load', 'Buying', 'Selling']}

    for n, result in results.items():
        table['Generation'][n] = result.get_total_generation()
        table['Load'][n] = (result.solu['l_sh']+result.solu['l_m']).sum()
        table['Buying'][n] = result.get_total_buy()
        table['Selling'][n] = result.get_total_sell()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/(N={N},T={T})power_table.csv')


def plot_demand(results, N, T):
    demand = np.array([result.solu['l_sh'] + result.solu['l_m'] for result in results.values()])
    shed0 = np.array([result.solu['l_sh_Non'] for result in results.values()])
    shed1 = np.array([result.solu['l_sh'] for result in results.values()])
    # Add the last value 
    demand = np.hstack((demand, demand[:, -1].reshape(-1, 1)))
    shed0 = np.hstack((shed0, shed0[:, -1].reshape(-1, 1)))
    shed1 = np.hstack((shed1, shed1[:, -1].reshape(-1, 1)))


    plt.rcParams['font.size'] = 12
    x = range(25)
    c = 'gray'
    for n in range(N):
        plt.figure(figsize=(5, 2))
        plt.step(x, shed1[n], where='post', color='gray', label='Shed After Trade')
        plt.fill_between(x, shed1[n], 0, step='post', color='gray', alpha=0.2)
        plt.step(x, shed0[n], where='post', color='red', label='Shed Before Trade')
        plt.step(x, demand[n], where='post', color='green', linestyle='--', label='Demand')
        #plt.fill_between(x, demand[n], 0, step='post', color=c[n], alpha=0.2)
        plt.xticks(np.linspace(0, 24, 13))
        plt.ylabel('(kW/h)')
        plt.xlabel('Time')
        plt.title(f'MG {n+1}')
        plt.legend(loc=[1.01, 0])
        plt.savefig(f'Figures/(N={N},T={T})shed(MG{n}).jpg', bbox_inches='tight', dpi=600)
        plt.close()

def compare_with_fullinfo(results, N, T):
    with open(f'Solution/(N={N},T={T})FullInfo-solu.pkl', 'rb') as handle:
        full_solu = pickle.load(handle)
    df = {
        'No-Trade Resilience': [result.solu['eta_r_Non'] for result in results.values()],
        'No-Trade Cost': [result.solu['eta_c_Non'] for result in results.values()],
        'Central Resilience': full_solu['eta_r'], 
        'Central Cost': full_solu['eta_c'], 
        'Distributed Resilience': [result.solu['eta_r'] for result in results.values()], 
        'Distributed Cost': [result.solu['eta_c'] for result in results.values()]}
    pd.DataFrame(df).to_csv(f'Results/(N={N},T={T})full_info_comparison.csv')