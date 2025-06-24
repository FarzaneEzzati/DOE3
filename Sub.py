import pickle
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from MGConfig import MG_config

class Sub:
    def __init__(self, data, trade, MIP=False):
        # Initialize
        self.config = MG_config(data)

        self.model = gp.Model(f'MG({self.config.mg_id})')
        self.model.setParam("OutputFlag", 0)

        # Buy/sell mode
        if MIP:
            self.x = self.model.addVars(self.config.HL, vtype=GRB.BINARY, name='u')
            self.model.setParam("Threads", 3)
            self.model.setParam("TimeLimit", 60)
            self.model.setParam("MIPGap", 0.01)
            self.model.setParam("NumericFocus", 1)  # Prioritize numerical stability
            self.model.setParam("FeasibilityTol", 1e-3)  # Increase feasibility tolerance
            self.model.setParam("Cuts", 2)
        else:
            self.x = self.model.addVars(self.config.HL, lb=0, ub=1, name='u')
        # Trading variables
        self.e_buy = self.model.addMVar((self.config.I, self.config.HL), name='e_buy')
        self.e_sell = self.model.addMVar((self.config.I, self.config.HL), name='e_sell')
        self.pi_buy = self.model.addMVar((self.config.I, self.config.HL), name='pi_buy')
        self.pi_sell = self.model.addMVar((self.config.I, self.config.HL), name='pi_sell')
        # Vars for admm
        stacked_trades = np.stack((self.e_buy, self.e_sell, self.pi_buy, self.pi_sell), axis=0)
        self.y = np.array([mvar.tolist() for mvar in stacked_trades], dtype=object)

        # Device variables
        self.g_pv = self.model.addMVar((self.config.S, self.config.HL), name='g_pv')
        self.g_dg =  self.model.addMVar((self.config.S, self.config.HL), name='g_dg')
        self.l_m =  self.model.addMVar((self.config.S, self.config.HL), name='l_m')
        self.r_c =  self.model.addMVar((self.config.S, self.config.HL), name='r_c')
        self.r_d =  self.model.addMVar((self.config.S, self.config.HL), name='r_d')
        self.e_l =  self.model.addMVar((self.config.S, self.config.HL), name='e_l')
        self.l_sh =  self.model.addMVar((self.config.S, self.config.HL), name='l_sh')
        # Resilience and financial benefits
        self.eta_r = self.model.addVar(ub=1, name='eta_r')
        self.eta_c = self.model.addVar(ub=1, name='eta_c')
        # Costs
        self.C_es = self.model.addVar(name='C_es')
        self.C_sh = self.model.addVar(name='C_ls')
        self.C_dg = self.model.addVar(name='C_dg')
        self.C_u = self.model.addVar(name='C_u')
        self.C_r = self.model.addVar(name='C_r')
        self.C_e = self.model.addVar(lb=-float('inf'), name='C_e')
        self.C_t = self.model.addVar(lb=-float('inf'), name='C_t')
        # Slacks
        self.e_buy_slack = self.model.addMVar((self.config.I, self.config.HL))
        self.e_sell_slack = self.model.addMVar((self.config.I, self.config.HL))
        self.pi_buy_slack_min = self.model.addMVar((self.config.I, self.config.HL))
        self.pi_sell_slack_min = self.model.addMVar((self.config.I, self.config.HL))
        self.pi_buy_slack_max = self.model.addMVar((self.config.I, self.config.HL))
        self.pi_sell_slack_max = self.model.addMVar((self.config.I, self.config.HL))

        self.g_pv_slack = self.model.addMVar((self.config.S, self.config.HL))
        self.g_dg_slack = self.model.addMVar((self.config.S, self.config.HL))
        self.l_m_slack = self.model.addMVar((self.config.S, self.config.HL))
        self.r_c_slack = self.model.addMVar((self.config.S, self.config.HL))
        self.r_d_slack = self.model.addMVar((self.config.S, self.config.HL))
        self.e_l_slack = self.model.addMVar((self.config.S, self.config.HL))


        self.eta_r_slack = self.model.addVar(ub=1)
        self.eta_c_slack = self.model.addVar(ub=1)

        # Constraints
        self.addConstraints()

        # Objective function
        self.fixed_obj = -(self.config.alpha *
                              (self.config.theta_r * self.eta_r +
                               self.config.theta_c * self.eta_c))
        self.model.setObjective(self.fixed_obj, sense=GRB.MINIMIZE)

        # Set lower bound for resilience and cost
        if trade:
            trade_off_constr = self.model.addConstr(self.e_buy.sum() + self.e_sell.sum() == 0, name='trade_off')
            self.model.optimize()

            if self.model.status == GRB.INFEASIBLE:
                self.show_infeasible_const()

            self.eta_r_Non = self.eta_r.x
            self.eta_c_Non = self.eta_c.x

            self.model.remove(trade_off_constr)

            self.model.addConstr(self.eta_r - self.eta_r_slack == self.config.tau_cutoff * self.eta_r_Non, name='res_improve')
            self.model.addConstr(self.eta_c - self.eta_c_slack == self.config.tau_cutoff * self.eta_c_Non, name='cost_improve')
            self.model.update()
        else:
            self.model.addConstr(self.e_buy.sum() + self.e_sell.sum() == 0, name='trade_off')
        self.model.update()
        self.obj_lagrangian = 0

    def addConstraints(self):

        # self trade
        self.model.addConstr(self.e_sell[self.config.id].sum() + self.e_buy[self.config.id].sum() == 0, name=f'self')
        # trade
        for j in self.config.i_index:
            for t in self.config.t_index:
                self.model.addConstr(self.e_buy[j, t] + self.e_buy_slack[j, t] == self.x[t] * self.config.M,
                                      name=f'buy_mode[{j},{t}]')
                self.model.addConstr(self.e_sell[j, t] + self.e_sell_slack[j, t] == (1 - self.x[t]) * self.config.M,
                                     name=f'sell_mode[{j},{t}]')
                self.model.addConstr(self.pi_buy[j, t] + self.pi_buy_slack_max[j, t] == self.config.pay_max * self.e_buy[j, t],
                                     name=f'buy_pay_max[{j},{t}]')
                self.model.addConstr(self.pi_buy[j, t] - self.pi_buy_slack_min[j, t] == self.config.pay_min * self.e_buy[j, t],
                                     name=f'buy_pay_min[{j},{t}]')
                self.model.addConstr(self.pi_sell[j, t] + self.pi_sell_slack_max[j, t] == self.config.pay_max * self.e_sell[j, t],
                                     name=f'sell_pay_max[{j},{t}]')
                self.model.addConstr(self.pi_sell[j, t] - self.pi_sell_slack_min[j, t] == self.config.pay_min * self.e_sell[j, t],
                                     name=f'sell_pay_min[{j},{t}]')
        # schedule
        for s in self.config.s_index:
            self.model.addConstr(self.e_l[s, 0] == self.config.es_capacity, name=f'e_l0')
            for t in self.config.t_index:
                self.model.addConstr(self.g_pv[s, t] + self.g_pv_slack[s, t] == self.config.pv_capacity * self.config.pv_hourly[t],
                                     name=f'g_pv_limit[{s},{t}]')
                self.model.addConstr(self.g_dg[s, t] + self.g_dg_slack[s, t] == self.config.dg_capacity * self.config.dg_effi,
                                      name=f'g_dg_limit[{s},{t}]')
                self.model.addConstr(self.r_c[s, t] + self.r_c_slack[s, t] == self.config.es_charge * self.config.es_capacity,
                                     name=f'es_doc[{s},{t}]')
                self.model.addConstr(self.r_d[s, t] + self.r_d_slack[s, t] == self.config.es_discha * self.config.es_capacity,
                                     name=f'es_dod[{s},{t}]')
                self.model.addConstr(self.e_l[s, t] + self.e_l_slack[s, t] == self.config.es_capacity,
                                     name=f'l limit[{s},{t}]')
                self.model.addConstr(self.l_sh[s, t] == self.config.load[s, t] - self.l_m[s, t],
                                     name='load_shed[{s},{t}]')
                self.model.addConstr(self.l_m[s, t] + self.r_c[s, t] + self.e_sell[:, t].sum() ==
                                     self.g_pv[s, t] + self.g_dg[s, t] + self.r_d[s, t] + self.e_buy[:, t].sum(),
                                     name=f'balance[{s},{t}]')
                if t < self.config.HL-1:
                    self.model.addConstr(
                        self.e_l[s, t + 1] == self.e_l[s, t] +
                        self.r_c[s, t] * self.config.es_charge -
                        self.r_d[s, t] / self.config.es_discha, name=f'flow[{s},{t}]')

        # costs
        self.model.addConstr(
            self.C_es == self.config.es_cost *
            sum(self.config.probs[s] * (self.r_c[s].sum() + self.r_d[s].sum())
                for s in self.config.s_index),
            name='C_es')
        self.model.addConstr(
            self.C_dg == self.config.dg_cost *
            sum(self.config.probs[s] * self.g_dg[s].sum()
                   for s in self.config.s_index),
            name='C_dg')
        self.model.addConstr(
            self.C_sh == self.config.shed_penalty *
            sum(self.config.probs[s] * self.l_sh[s].sum()
                for s in self.config.s_index),
            name='C_ls')
        self.model.addConstr(
            self.C_u == (1 - self.config.usrr) *
            self.config.utility_cost * (self.e_buy.sum() + self.e_sell.sum()),
            name='C_u')
        self.model.addConstr(
            self.C_e == (1 - self.config.fsrr) * self.pi_buy.sum() - self.pi_sell.sum(),
            name='C_e')
        self.model.addConstr(
            self.C_r == self.config.e_load * sum(self.config.probs[s] * self.l_m[s].sum()
                                                 for s in self.config.s_index), name='C_r')
        self.model.addConstr(
            self.C_t == self.C_es + self.C_dg + self.C_sh + self.C_u + self.C_e - self.C_r, name='C_t')
        # Energy equity
        self.model.addConstr(
            self.eta_r == np.sum(self.config.probs[s] * self.l_m[s].sum() / self.config.load[s].sum()
                                 for s in range(self.config.S)),
            name='eta_r')
        self.model.addConstr(
            self.eta_c == 1 - (self.C_t - self.config.C_t_min) / (self.config.C_t_max - self.config.C_t_min),
            name='eta_c')
        self.model.update()

    def solve_with(self, yhat, l_y, rho_y, z, l_z, rho_z):
        # yhat part
        y_minus_yhat = self.y - yhat
        lag_y = l_y * y_minus_yhat + 0.5 * rho_y * y_minus_yhat ** 2
        lag_y_sum = lag_y.sum()
        # x part
        x_minus_z = self.x.values() - z
        lag_z = l_z * x_minus_z + 0.5 * rho_z * x_minus_z ** 2
        lag_z_sum = lag_z.sum()
        # Objective
        self.obj_lagrangian = lag_y_sum + lag_z_sum
        self.model.setObjective(1000 * self.fixed_obj + self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return y is model optimal or timed out, o.w. interrupt
        if self.model.Status == GRB.OPTIMAL or self.model.SolCount > 0:
            y_opt = np.stack([self.e_buy.x,
                          self.e_sell.x,
                          self.pi_buy.x,
                          self.pi_sell.x])
            u_opt = np.array([x.x for x in self.x.values()])
            return y_opt, u_opt
        else:
            print(self.model.Status)
            print('Separation stopped.')

    def show_infeasible_const(self):
        self.model.computeIIS()
        for c in self.model.getConstrs():
            if c.IISConstr:
                print('infeasible constr: ', c.ConstrName)

    def save_solutions(self, yhat, l_y, rho_y, z, l_z, rho_z, name):
        _ = self.solve_with(yhat=yhat, l_y=l_y, rho_y=rho_y, z=z, l_z=l_z, rho_z=rho_z)

        vars_to_save = {
            'g_pv': self.g_pv.x,
            'g_dg': self.g_dg.x,
            'l_m': self.l_m.x,
            'r_c': self.r_c.x,
            'r_d': self.r_d.x,
            'e_l': self.e_l.x,
            'l_sh': self.l_sh.x,
            'e_sell': self.e_sell.x,
            'e_buy': self.e_buy.x,
            'pi_sell': self.pi_sell.x,
            'pi_buy': self.pi_buy.x,
            'x': np.array([x.x for x in self.x.values()]),
            'eta_r': self.eta_r.x,
            'eta_c': self.eta_c.x,
            'tau_cutoff': self.config.tau_cutoff
        }
        costs_to_save = {
            'C_es': self.C_es.x,
            'C_ls': self.C_sh.x,
            'C_dg': self.C_dg.x,
            'C_u': self.C_u.x,
            'C_e': self.C_e.x,
            'C_t': self.C_t.x}

        with open(f'Results/{name}_MG({self.config.mg_id}).pkl', 'wb') as handle:
            pickle.dump([vars_to_save, costs_to_save], handle)




