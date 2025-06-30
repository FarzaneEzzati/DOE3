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
        return self.costs['C_es'] + self.costs['C_ls'] + self.costs['C_dg'] + self.costs['C_u'] + self.costs['C_e']
    def get_payments(self):
        return self.vars['pi_buy'].sum() - self.vars['pi_sell'].sum()
    def get_utility_cost(self):
        return (self.vars['e_buy'].sum() + self.vars['e_sell'].sum()) * self.data_mg['u_cost']
    def get_kpi(self):
        # Connection rate
        did_buy = self.vars['e_buy'].sum(axis=0) == 0
        did_shed = (np.average(self.vars['l_sh'], weights=self.probs, axis=0) >=
                    0.5 * np.average(self.vars['l_m'], weights=self.probs, axis=0))
        CR = 100 * (1 - np.sum(did_buy & did_shed) / self.HL)

        # Renewable Energy Access Rate
        generated_pv = np.average(self.vars['g_pv'], weights=self.probs, axis=0).sum()
        demanded_load = np.average(self.vars['l_m'], weights=self.probs, axis=0).sum()
        REAR = 100 *  generated_pv/demanded_load

        # CO2 emission, 0.4–0.5 kg CO₂/kWh is grid baseline
        generated_dg = np.average(self.vars['g_dg'], weights=self.probs, axis=0).sum()
        served_load = np.average(self.vars['l_m'], weights=self.probs, axis=0).sum()
        dg_CO2 =  generated_dg * self.CO2
        grid_CO2 = served_load * 0.45
        CO2 = 100 * abs(dg_CO2 - grid_CO2)/grid_CO2

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



def plot_trade(results_after, name_to_save):
    for key in results_after.keys():
        y = (results_after[key].vars['e_buy'] - results_after[key].vars['e_sell']).sum(axis=0)
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


def plot_load_served(results_after, quantile, name_to_save):
    for key in results_after.keys():
        y1 = results_after[key].vars['l_m']
        y1 = np.hstack((y1, y1[:, -1].reshape(-1, 1)))
        x = np.arange(y1.shape[1])

        y2 = results_after[key].vars['l_sh']
        y2 = np.hstack((y2, y2[:, -1].reshape(-1, 1)))

        plt.figure(figsize=(5, 4))
        plt.step(x=x, y=y1[quantile], where='post', label=f'Served Load', color='green')
        plt.step(x=x, y=y1[quantile]+y2[quantile], where='post', label=f'Total Load', color='red', linestyle=':')
        plt.ylabel('Load (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.savefig(f'Figures/{name_to_save}load({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_trade(results_after: Dict[int, MGResults], name_to_save):
    buy = np.stack([result.vars['e_buy'] for result in results_after.values()])
    sell = np.stack([result.vars['e_sell'] for result in results_after.values()])
    buy_sell = buy - sell
    buy_sell = np.sum(buy_sell, axis=1)
    buy_sell = np.concatenate((buy_sell, buy_sell[:, -1][:, np.newaxis]), axis=1)

    plt.rcParams['font.size'] = 12
    for n in results_after.keys():
        x, y, c = range(25), buy_sell[n], ['orange', 'red', 'blue']
        fig = plt.figure()
        plt.step(x, y, where='post', color=c[n], label=f'MG {n}')
        plt.fill_between(x, y, 0, step='post', color=c[n], alpha=0.3)
        plt.xticks(np.linspace(0, 24, 13))
        plt.plot([0, 24], [0, 0], ':', color='black')
        plt.ylabel('(kW/h)')
        plt.xlabel('Time')
        plt.ylim([-60, 50])
        plt.title(f'MG {n+1}')
        plt.savefig(f'Figures/{name_to_save}price(MG{n}).jpg', bbox_inches='tight', dpi=600)
        plt.close()

def get_cost_table(results_before, results_after: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Cost Before Trade ($)',
                                                        'Cost After Trade ($)',
                                                        'Utility Fee ($)',
                                                        'Sell Paid ($)',
                                                        'Buy Paid ($)',
                                                        'Utility Subsidy Usage ($)',
                                                        'Purchase Subsidy Usage ($)',
                                                        'Net Cost After Trade']}
    for i in results_before.keys():
        table['Cost Before Trade ($)'][i] = results_before[i].get_opr_cost()
        table['Cost After Trade ($)'][i] = results_after[i].get_opr_cost()
        table['Utility Fee ($)'][i] = results_after[i].get_utility_cost()
        table['Sell Paid ($)'][i] = results_after[i].vars['pi_sell'].sum()
        table['Buy Paid ($)'][i] = results_after[i].vars['pi_buy'].sum()
        table['Utility Subsidy Usage ($)'][i] = results_after[i].get_utility_subsidy_usage()
        table['Purchase Subsidy Usage ($)'][i] = results_after[i].get_purchase_subsidy_usage()
        table['Net Cost After Trade'][i] = results_after[i].get_net_cost()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}cost_table.csv')


def get_metric_table(results_before: Dict[int, MGResults], results_after: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Resilience Before Trade',
                                                'Cost Before Trade',
                                                'Resilience After Trade',
                                                'Cost After Trade']}
    for i in results_before.keys():
        table['Resilience Before Trade'][i] = results_before[i].vars['eta_r']
        table['Resilience After Trade'][i] = results_after[i].vars['eta_r']
        table['Cost Before Trade'][i] = results_before[i].vars['eta_c']
        table['Cost After Trade'][i] = results_after[i].vars['eta_c']
    # adding system wide
    table = {key: np.append(val, np.average(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}metric_table.csv')


def get_kpi_table(results_after: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Connection Rate', 'Renewable Energy Access Rate',
                                                'CO2 Emission Reduction', 'Renewable Energy Mandate']}
    for i in results_after.keys():
        CR, REAR, CO2, REM = results_after[i].get_kpi()
        table['Connection Rate'][i] = CR
        table['Renewable Energy Access Rate'][i] = REAR
        table['CO2 Emission Reduction'][i] = CO2
        table['Renewable Energy Mandate'][i] = REM
    table = {key: np.append(val, np.average(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}kpi_table.csv')


def get_power_table(results_after: Dict[int, MGResults], num_mgs, name_to_save):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Generation', 'Average Load', 'Buying', 'Selling']}

    for i in results_after.keys():
        table['Generation'][i] = results_after[i].get_total_generation()
        table['Average Load'][i] = results_after[i].get_avg_load()
        table['Buying'][i] = results_after[i].get_total_buy()
        table['Selling'][i] = results_after[i].get_total_sell()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Results/{name_to_save}power_table.csv')

