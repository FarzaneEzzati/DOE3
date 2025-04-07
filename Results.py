import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class MGResults:
    def __init__(self, file_name, data_mg):
        with open(f'Solutions/{file_name}.pkl', 'rb') as handle:
            self.final_l, self.final_z, self.final_rho, \
                self.g_pv, self.g_dg, self.x, self.r_c, self.r_d, self.el, self.ls, \
                self.e_sell, self.e_buy, self.pi_sell, self.pi_buy, self.u, \
                self.eta_r, self.eta_c, self.eta_r_Non, self.eta_c_Non, \
                self.C_es, self.C_ls, self.C_dg, self.C_u, self.C_e, self.C_t = pickle.load(handle)
        self.probs = data_mg['probs']
        self.l = data_mg['l_s']
        self.HL = self.l[0].shape[0]
        self.CO2 = data_mg['CO2']
        self.data_mg = data_mg

        self.fsrr = data_mg['fsr'] * data_mg['sv']  # financial support rate
        self.usrr = data_mg['usr'] * data_mg['sv']  # utility support rate
    def get_opr_cost(self):
        return self.C_es + self.C_ls + self.C_dg
    def get_net_cost(self):
        return self.C_es + self.C_ls + self.C_dg + self.C_u + self.C_e
    def get_payments(self):
        return np.sum(self.pi_buy - self.pi_sell)
    def get_utility_cost(self):
        return np.sum(self.e_buy + self.e_sell) * self.data_mg['u_cost']
    def get_kpi(self):
        # Connection rate
        did_buy = self.e_buy.sum(axis=0) == 0
        did_shed = (np.average(self.ls, weights=self.probs, axis=0) >=
                    0.5 * np.average(self.l, weights=self.probs, axis=0))
        CR = 100 * (1 - np.sum(did_buy & did_shed) / self.HL)

        # Renewable Energy Access Rate
        generated_pv = np.average(self.g_pv, weights=self.probs, axis=0).sum()
        demanded_load = np.average(self.l, weights=self.probs, axis=0).sum()
        REAR = 100 *  generated_pv/demanded_load

        # CO2 emission, 0.4–0.5 kg CO₂/kWh is grid baseline
        generated_dg = np.average(self.g_dg, weights=self.probs, axis=0).sum()
        served_load = np.average(self.x, weights=self.probs, axis=0).sum()
        dg_CO2 =  generated_dg * self.CO2
        grid_CO2 = served_load * 0.45
        CO2 = 100 * abs(dg_CO2 - grid_CO2)/grid_CO2

        # Renewable Energy Mandate
        REM = 100 * generated_pv / (generated_pv + generated_dg)

        return CR, REAR, CO2, REM
    def get_utility_subsidy_usage(self):
        return np.sum(self.e_buy + self.e_sell) * self.data_mg['u_cost'] * self.usrr
    def get_purchase_subsidy_usage(self):
        return np.sum(self.pi_buy) * self.fsrr


def plot_trade(results_after, identifier):
    for key in results_after.keys():
        y = (results_after[key].e_buy - results_after[key].e_sell).sum(axis=0)
        y = np.append(y, y[-1])
        x = np.arange(y.shape[0])

        plt.figure(figsize=(5, 4))
        plt.step(x=x, y=y, where='post')
        plt.axhline(0, color='red', linestyle=':')
        plt.ylabel('Energy Traded (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.savefig(f'Solutions/{identifier}trade({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def plot_load_served(results_after, quantile, identifier):
    for key in results_after.keys():
        y1 = results_after[key].x
        y1 = np.hstack((y1, y1[:, -1].reshape(-1, 1)))
        x = np.arange(y1.shape[1])

        y2 = results_after[key].ls
        y2 = np.hstack((y2, y2[:, -1].reshape(-1, 1)))

        plt.figure(figsize=(5, 4))
        plt.step(x=x, y=y1[quantile], where='post', label=f'Served Load', color='green')
        plt.step(x=x, y=y1[quantile]+y2[quantile], where='post', label=f'Total Load', color='red', linestyle=':')
        plt.ylabel('Load (kWh)')
        plt.xlabel('Time (Hour)')
        plt.xlim((0, 24))
        plt.xticks(range(0, 25, 2))
        plt.legend()
        plt.savefig(f'Solutions/{identifier}load({key}).jpg', dpi=300, bbox_inches='tight')
        plt.close()


def get_cost_table(results_before, results_after, num_mgs, identifier):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Cost Before Trade ($)',
                                                        'Cost After Trade ($)',
                                                        'Utility Fee ($)',
                                                        'Payments ($)',
                                                        'Utility Subsidy Usage ($)',
                                                        'Purchase Subsidy Usage ($)',
                                                        'Net Cost After Trade']}
    for i in range(num_mgs):
        table['Cost Before Trade ($)'][i] = results_before[i].get_opr_cost()
        table['Cost After Trade ($)'][i] = results_after[i].get_opr_cost()
        table['Utility Fee ($)'][i] = results_after[i].get_utility_cost()
        table['Payments ($)'][i] = results_after[i].get_payments()
        table['Utility Subsidy Usage ($)'][i] = results_after[i].get_utility_subsidy_usage()
        table['Purchase Subsidy Usage ($)'][i] = results_after[i].get_purchase_subsidy_usage()
        table['Net Cost After Trade'][i] = results_after[i].get_net_cost()
    table = {key: np.append(val, np.sum(val)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Solutions/{identifier}cost_table.csv')


def get_metric_table(results_before, results_after, num_mgs, data_alpha, identifier):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Resilience Before Trade',
                                                'Cost Before Trade',
                                                'Resilience After Trade',
                                                'Cost After Trade']}
    for i in range(num_mgs):
        table['Resilience Before Trade'][i] = results_before[i].eta_r
        table['Resilience After Trade'][i] = results_after[i].eta_r
        table['Cost Before Trade'][i] = results_before[i].eta_c
        table['Cost After Trade'][i] = results_after[i].eta_c
    table = {key: np.append(val, np.average(val, weights=data_alpha)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Solutions/{identifier}metric_table.csv')


def get_kpi_table(results_after, num_mgs, data_alpha, identifier):
    indices = [f'Community {i}' for i in range(num_mgs)] + ['System']
    table = {key: np.zeros(num_mgs) for key in ['Connection Rate', 'Renewable Energy Access Rate',
                                                'CO2 Emission Reduction', 'Renewable Energy Mandate']}
    for i in range(num_mgs):
        CR, REAR, CO2, REM = results_after[i].get_kpi()
        table['Connection Rate'][i] = CR
        table['Renewable Energy Access Rate'][i] = REAR
        table['CO2 Emission Reduction'][i] = CO2
        table['Renewable Energy Mandate'][i] = REM
    table = {key: np.append(val, np.average(val, weights=data_alpha)) for key, val in table.items()}
    pd.DataFrame(table, index=indices).to_csv(f'Solutions/{identifier}kpi_table.csv')
