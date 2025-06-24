import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from MGConfig import MG_config


class FullModel:
    def __init__(self, mg_data: dict, cems_data: dict):
        self.configs = [MG_config(data) for data in mg_data.values()]
        self.I = self.configs[0].I
        self.HL = self.configs[0].HL
        self.S = self.configs[0].S
        self.TFS = cems_data['TFS']
        self.TUS = cems_data['TUS']

        self.tau_cutoff = [config.tau_cutoff for config in self.configs]

        self.model = gp.Model('FullModel')
        self.model.setParam('OutputFlag', 0)

        # Buy/sell mode
        self.u = self.model.addMVar((self.I, self.HL), vtype=GRB.BINARY, name='u')
        # Trading variables
        self.e_buy = self.model.addMVar((self.I, self.I, self.HL), name='e_buy')  # energy traded
        self.e_sell = self.model.addMVar((self.I, self.I, self.HL), name='e_sell')  # energy traded
        self.pi_buy = self.model.addMVar((self.I, self.I, self.HL), name='pi_buy')  # energy traded
        self.pi_sell = self.model.addMVar((self.I, self.I, self.HL), name='pi_sell')  # energy traded
        # Device variables
        self.g_pv = self.model.addMVar((self.I, self.S, self.HL), name='g_pv')  # generation by pv devices
        self.g_dg = self.model.addMVar((self.I, self.S, self.HL), name='g_dg')  # generation by dg devices
        self.l_m = self.model.addMVar((self.I, self.S, self.HL), name='l_m')  # load met
        self.r_c = self.model.addMVar((self.I, self.S, self.HL), name='r_c')  # charging amount
        self.r_d = self.model.addMVar((self.I, self.S, self.HL), name='r_d')  # discharging amount
        self.e_l = self.model.addMVar((self.I, self.S, self.HL), name='e_l')  # energy level at es
        self.l_sh = self.model.addMVar((self.I, self.S, self.HL), name='l_sh')    # load shed
        # Resilience and financial benefits
        self.eta_r = self.model.addMVar(self.I, ub=1, name='eta_r')
        self.eta_c = self.model.addMVar(self.I, ub=1, name='eta_c')
        # Costs
        self.C_es = self.model.addMVar(self.I, name='C_es')
        self.C_sh = self.model.addMVar(self.I, name='C_ls')
        self.C_dg = self.model.addMVar(self.I, name='C_dg')
        self.C_u = self.model.addMVar(self.I, name='C_u')
        self.C_r = self.model.addMVar(self.I, name='C_r')
        self.C_e = self.model.addMVar(self.I, lb=-float('inf'), name='C_e')
        self.C_t = self.model.addMVar(self.I, lb=-float('inf'), name='C_t')
        self.purchase_subsidy = [0 for _ in range(self.I)]

        for i in range(self.I):
            ####### available power of devices
            max_r_c = self.configs[i].es_charge * self.configs[i].es_capacity
            max_r_d = self.configs[i].es_discha * self.configs[i].es_capacity
            max_g_dg = self.configs[i].dg_capacity * self.configs[i].dg_effi

            ####### trades
            self.model.addConstr(self.e_sell[i, i].sum() + self.e_buy[i, i].sum() == 0)
            # energy equity
            eta_r = sum(self.configs[i].probs[s] * self.l_m[i, s].sum() / self.configs[i].load[s].sum()
                        for s in range(self.S))
            eta_c = 1 - (self.C_t[i] - self.configs[i].C_t_min) / (self.configs[i].C_t_max - self.configs[i].C_t_min)
            self.model.addConstr(self.eta_r[i] == eta_r, name=f'eta_r{i}')
            self.model.addConstr(self.eta_c[i] == eta_c, name=f'eta_c{i}')
            # hourly trade
            for j in range(self.I):
                for t in range(self.HL):
                    self.model.addConstr(self.e_buy[i, j, t] <= self.u[i, t] * self.configs[i].M,
                                         name=f'buy_mode[{i},{j},{t}]')
                    self.model.addConstr(self.e_sell[i, j, t] <= (1 - self.u[i, t]) * self.configs[i].M,
                                         name=f'sell_mode[{i},{j},{t}]')
                    # trade limits
                    max_buy_payment = self.configs[i].pay_max * self.e_buy[i, j, t]
                    min_buy_payment = self.configs[i].pay_min * self.e_buy[i, j, t]
                    self.model.addConstr(self.pi_buy[i, j, t] <= max_buy_payment,
                                         name=f'max_buy_payment[{i},{j},{t}]')
                    self.model.addConstr(self.pi_buy[i, j, t] >= min_buy_payment,
                                         name=f'min_buy_payment[{i},{j},{t}]')

                    max_sell_payment = self.configs[i].pay_max * self.e_sell[i, j, t]
                    min_sell_payment = self.configs[i].pay_min * self.e_sell[i, j, t]
                    self.model.addConstr(self.pi_sell[i, j, t] <= max_sell_payment,
                                         name=f'max_sell_payment[{i},{j},{t}]')
                    self.model.addConstr(self.pi_sell[i, j, t] >= min_sell_payment,
                                         name=f'min_sell_payment[{i},{j},{t}]')

                    # market constraints
                    self.model.addConstr(self.e_buy[i, j, t] - self.e_sell[j, i, t] == 0,
                                         name=f'e_clear[{i},{j},{t}]')
                    self.model.addConstr(self.pi_buy[i, j, t] - self.pi_sell[j, i, t] == 0,
                                         name=f'pi_clear[{i},{j},{t}]')

            ####### scheduling
            for s in range(self.S):
                self.model.addConstr(self.e_l[i, s, 0] == self.configs[i].es_capacity)
                for t in range(self.HL):
                    max_g_pv = self.configs[i].pv_capacity * self.configs[i].pv_hourly[t]
                    self.model.addConstr(self.g_pv[i, s, t] <= max_g_pv, name=f'g_pv_limit[{i},{s},{t}]')
                    self.model.addConstr(self.g_dg[i, s, t] <= max_g_dg, name=f'g_dg_limit[{i},{s},{t}]')

                    self.model.addConstr(self.r_c[i, s, t] <= max_r_c, name=f'es_doc[{i},{s},{t}]')
                    self.model.addConstr(self.r_d[i, s, t] <= max_r_d, name=f'es_dod[{i},{s},{t}]')
                    self.model.addConstr(self.e_l[i, s, t] <= self.configs[i].es_capacity, name=f'e_l_max[{i},{s},{t}]')

                    load_shed = self.configs[i].load[s, t] - self.l_m[i, s, t]
                    self.model.addConstr(self.l_sh[i, s, t] == load_shed, name=f'load_shed[{i},{s},{t}]')

                    power_input = self.g_pv[i, s, t] + self.g_dg[i, s, t] + self.r_d[i, s, t] + self.e_buy[i, :, t].sum()
                    power_output = self.l_m[i, s, t] + self.r_c[i, s, t] + self.e_sell[i, :, t].sum()
                    self.model.addConstr(power_output == power_input, name=f'balance[{i},{s},{t}]')

                    if t < self.HL - 1:
                        es_level_change = self.e_l[i, s, t] + self.r_c[i, s, t] * self.configs[i].es_charge - \
                                          self.r_d[i, s, t] / self.configs[i].es_discha
                        self.model.addConstr(self.e_l[i, s, t + 1] == es_level_change, name=f'flow[{i},{s},{t}]')

            ####### costs
            avg_charge_discharge = sum((self.r_c[i, s].sum() + self.r_d[i, s].sum()) * self.configs[i].probs[s]
                                       for s in range(self.S))
            C_es = self.configs[i].es_cost * avg_charge_discharge
            self.model.addConstr(self.C_es[i] == C_es)

            avg_dg = sum(self.g_dg[i, s].sum() * self.configs[i].probs[s] for s in range(self.S))
            C_dg = self.configs[i].dg_cost * avg_dg
            self.model.addConstr(self.C_dg[i] == C_dg)

            avg_load_shed = sum(self.l_sh[i, s].sum() * self.configs[i].probs[s] for s in range(self.S))
            C_sh = self.configs[i].shed_penalty * avg_load_shed
            self.model.addConstr(self.C_sh[i] == C_sh)

            total_trade = self.e_buy[i].sum() + self.e_sell[i].sum()
            C_u = (1 - self.configs[i].usrr) * self.configs[i].utility_cost * total_trade
            self.model.addConstr(self.C_u[i] == C_u)

            C_e = (1 - self.configs[i].fsrr) * self.pi_buy[i].sum() - self.pi_sell[i].sum()
            self.model.addConstr(self.C_e[i] == C_e)

            C_r = self.configs[i].e_load * sum(self.configs[i].probs[s] * self.l_m[i, s].sum()
                                                 for s in self.configs[i].s_index)
            self.model.addConstr(self.C_r[i] == C_r)

            C_t = self.C_es[i] + self.C_dg[i] + self.C_sh[i] + self.C_u[i] + self.C_e[i] - self.C_r[i]
            self.model.addConstr(self.C_t[i] == C_t)

        # Subsidies
        purchase_subsidy_used = sum(self.pi_buy[i].sum() * self.configs[i].fsrr for i in range(self.I))
        utility_subsidy_used = sum(self.configs[i].utility_cost * self.configs[i].usrr *
                                   (self.e_buy[i].sum() + self.e_sell[i].sum())
                                   for i in range(self.I))
        self.model.addConstr(purchase_subsidy_used <= self.TFS, name='TFS')
        self.model.addConstr(utility_subsidy_used <= self.TUS, name='TUS')

        # Objective
        self.objective = sum((self.configs[i].alpha *
                              (self.configs[i].theta_r * self.eta_r[i] +
                               self.configs[i].theta_c * self.eta_c[i]) for i in range(self.I)))
        self.model.setObjective(self.objective, sense=GRB.MAXIMIZE)

        # Trade off baseline
        trade_off_constr = self.model.addConstr(self.e_buy.sum() + self.e_sell.sum() == 0, name='trade_off')
        self.model.optimize()

        self.eta_r_Non = self.eta_r.x
        self.eta_c_Non = self.eta_c.x
        self.C_t_Non = self.C_t.x

        self.model.remove(trade_off_constr)

        for i in range(self.I):
            self.model.addConstr(self.eta_r[i] >= self.tau_cutoff[i] * self.eta_r_Non[i])
            self.model.addConstr(self.eta_c[i] >= self.tau_cutoff[i] * self.eta_c_Non[i])
        self.model.update()




