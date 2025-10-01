import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict



class MGResults:
    def __init__(self, name, data_mg):
        with open(f'Results/{name}.pkl', 'rb') as handle:
            self.vars, self.costs = pickle.load(handle)
        self.probs = data_mg['probs']
        self.l_s = data_mg['l_s']
        self.HL = self.l_s[0].shape[0]
        self.CO2 = data_mg['CO2']
        self.data_mg = data_mg
        self.fsrr = data_mg['fsrr']
        self.usrr = data_mg['usrr']
    def get_opr_cost(self):
        return self.costs['C_es'] + self.costs['C_ls'] + self.costs['C_dg']
    def get_net_cost(self):
        return self.costs['C_t'] 
    def get_payments(self):
        return self.vars['pi_buy'].sum() - self.vars['pi_sell'].sum()
    def get_utility_cost(self):
        return (self.vars['e_buy'].sum() + self.vars['e_sell'].sum()) * self.data_mg['u_cost']
    def get_kpi(self):
        # Connection rate
        did_buy = np.average(self.vars['e_buy'], weights=self.probs, axis=0) == 0
        did_shed = (np.average(self.vars['l_sh'], weights=self.probs, axis=0) >=
                    0.7 * np.average(self.vars['l_m'], weights=self.probs, axis=0))
        CR = 100 * (1 - np.sum(did_shed) / self.HL)

        # Renewable Energy Access Rate
        generated_pv = np.average(self.vars['g_pv'], weights=self.probs, axis=0).sum()
        demanded_load = np.average(self.vars['l_m'], weights=self.probs, axis=0).sum()
        REAR = 100 *  generated_pv/demanded_load

        # CO2 emission, 0.4–0.5 kg CO₂/kWh is grid baseline
        generated_dg = np.average(self.vars['g_dg'], weights=self.probs, axis=0).sum()
        served_load = np.average(self.vars['l_m'], weights=self.probs, axis=0).sum()
        dg_CO2 =  generated_dg * self.CO2
        grid_CO2 = served_load * 0.45
        CO2 = 100 * (grid_CO2 - dg_CO2)/grid_CO2

        # Renewable Energy Mandate
        REM = 100 * generated_pv / (generated_pv + generated_dg)

        return CR, REAR, CO2, REM
    def get_utility_subsidy_usage(self):
        return self.get_utility_cost() * self.usrr
    def get_purchase_subsidy_usage(self):
        return np.sum(self.vars['pi_buy']) * self.fsrr
    def get_total_generation(self):
        pv = np.average(self.vars['g_pv'], weights=self.probs, axis=0).sum()
        dg = np.average(self.vars['g_dg'], weights=self.probs, axis=0).sum()
        return pv + dg
    def get_avg_load(self):
        return np.average(self.l_s, weights=self.probs, axis=0).sum()
    def get_total_buy(self):
        return self.vars['e_buy'].sum()
    def get_total_sell(self):
        return self.vars['e_sell'].sum()
    def get_buy(self):
        return self.vars['pi_buy']
    def get_sell(self):
        return self.vars['pi_sell']



def plot_trade(results1, name_to_save):
    for key in results1.keys():
        y = (results1[key].vars['e_buy'] - results1[key].vars['e_sell']).sum(axis=0)
        y = np.append(y, y[-1])
        x = np.arange(y.shape[0])

        plt.figure(figsize=(5, 4))
        plt.step(x=x, y=y, where='post')
        plt.axhline(0, color='red', linestyle=':')
        plt.ylabel('Energy Traded (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.savefig(f'Figures/{name_to_save}trade({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_load_served(results1, quantile, name_to_save):
    for key in results1.keys():
        y1 = results1[key].vars['l_m']
        y1 = np.hstack((y1, y1[:, -1].reshape(-1, 1)))
        x = np.arange(y1.shape[1])

        y2 = results1[key].vars['l_sh']
        y2 = np.hstack((y2, y2[:, -1].reshape(-1, 1)))

        plt.figure(figsize=(5, 2))
        plt.step(x=x, y=y1[quantile], where='post', label=f'Served Load', color='green')
        plt.step(x=x, y=y1[quantile]+y2[quantile], where='post', label=f'Total Load', color='red', linestyle=':')
        plt.ylabel('Load (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.savefig(f'Figures/{name_to_save}load({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_trade(results1: Dict[int, MGResults], name_to_save):
    buy = np.stack([result.vars['e_buy'] for result in results1.values()])
    sell = np.stack([result.vars['e_sell'] for result in results1.values()])
    buy_sell = buy - sell
    buy_sell = np.sum(buy_sell, axis=1)
    buy_sell = np.concatenate((buy_sell, buy_sell[:, -1][:, np.newaxis]), axis=1)
    y_min, y_max = np.min(buy_sell), np.max(buy_sell)
    c = ['orange', 'red', 'blue']
    plt.rcParams['font.size'] = 12
    for n in results1.keys():
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
        plt.savefig(f'Figures/{name_to_save}trade(MG{n}).jpg', bbox_inches='tight', dpi=600)
        plt.close()


def get_cost_table(results0, results1: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Cost Before Trade ($)',
                                                        'Cost After Trade ($)',
                                                        'Utility Fee ($)',
                                                        'Sell Paid ($)',
                                                        'Buy Paid ($)',
                                                        'Utility Subsidy Usage ($)',
                                                        'Purchase Subsidy Usage ($)',
                                                        'Net Cost After Trade']}
    for i in results0.keys():
        table['Cost Before Trade ($)'][i] = results0[i].get_opr_cost()
        table['Cost After Trade ($)'][i] = results1[i].get_opr_cost()
        table['Utility Fee ($)'][i] = results1[i].get_utility_cost()
        table['Sell Paid ($)'][i] = results1[i].vars['pi_sell'].sum()
        table['Buy Paid ($)'][i] = results1[i].vars['pi_buy'].sum()
        table['Utility Subsidy Usage ($)'][i] = results1[i].get_utility_subsidy_usage()
        table['Purchase Subsidy Usage ($)'][i] = results1[i].get_purchase_subsidy_usage()
        table['Net Cost After Trade'][i] = results1[i].get_net_cost()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}cost_table.csv')


def get_metric_table(results0: Dict[int, MGResults], results1: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Resilience Before Trade',
                                                'Cost Before Trade',
                                                'Resilience After Trade',
                                                'Cost After Trade']}
    for i in results0.keys():
        table['Resilience Before Trade'][i] = results0[i].vars['eta_r']
        table['Resilience After Trade'][i] = results1[i].vars['eta_r']
        table['Cost Before Trade'][i] = results0[i].vars['eta_c']
        table['Cost After Trade'][i] = results1[i].vars['eta_c']
    # adding system wide
    table = {key: np.append(val, np.average(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}metric_table.csv')


def get_kpi_table(results1: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Connection Rate', 'Renewable Energy Access Rate',
                                                'CO2 Emission Reduction', 'Renewable Energy Mandate']}
    for i in results1.keys():
        CR, REAR, CO2, REM = results1[i].get_kpi()
        table['Connection Rate'][i] = CR
        table['Renewable Energy Access Rate'][i] = REAR
        table['CO2 Emission Reduction'][i] = CO2
        table['Renewable Energy Mandate'][i] = REM
    table = {key: np.append(val, np.average(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}kpi_table.csv')


def get_power_table(results1: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Generation', 'Average Load', 'Buying', 'Selling']}

    for i in results1.keys():
        table['Generation'][i] = results1[i].get_total_generation()
        table['Average Load'][i] = results1[i].get_avg_load()
        table['Buying'][i] = results1[i].get_total_buy()
        table['Selling'][i] = results1[i].get_total_sell()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}power_table.csv')


def plot_demand(results0: Dict[int, MGResults], results1: Dict[int, MGResults], name_to_save):
    demand = np.array([np.average(r.l_s, weights=r.probs, axis=0) for r in results0.values()])
    shed0 = np.array([np.average(r.vars['l_sh'], weights=r.probs, axis=0) for r in results0.values()])
    shed1 = np.array([np.average(r.vars['l_sh'], weights=r.probs, axis=0) for r in results1.values()])
    demand = np.concatenate((demand, demand[:, -1][:, np.newaxis]), axis=1)
    shed0 = np.concatenate((shed0, shed0[:, -1][:, np.newaxis]), axis=1)
    shed1 = np.concatenate((shed1, shed1[:, -1][:, np.newaxis]), axis=1)

    plt.rcParams['font.size'] = 12
    x = range(25)
    c = 'gray'
    for n in results1.keys():
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
        plt.savefig(f'Figures/{name_to_save}shed(MG{n}).jpg', bbox_inches='tight', dpi=600)
        plt.close()
