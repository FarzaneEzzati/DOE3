"""
The subproblems for ADMM are defined here.
"""

import pickle
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

status = {1: 'LOADED',
         2: 'OPTIMAL',
         3: 'INFEASIBLE',
         4: 'INF_OR_UNBD',
         5: 'UNBOUNDED',
         6: 'CUTOFF',
         7: 'ITERATION_LIMIT',
         8: 'NODE_LIMIT',
         9: 'TIME_LIMIT',
         10: 'SOLUTION_LIMIT',
         11: 'INTERRUPTED',
         12: 'NUMERIC',
         13: 'SUBOPTIMAL',
         14: 'INPROGRESS',
         15: 'USER_OBJ_LIMIT'}


class LowerLevelModel:
    def __init__(self, data, trade, MIP, solver="Default"):
        # initialize
        I = data['I']    # number of microgrids
        id = data['id']  # id of microgrid
        HL = data['HL']  # horizon len
        S = data['S']    # number of scenarios
        fsrr = data['fsr'] * data['sv']  # financial support rate
        usrr = data['usr'] * data['sv']  # utility support rate
        C_t_min = data['C_t_min']
        C_t_max = data['C_t_max']

        self.I = I
        self.HL = HL
        self.S = S
        self.alpha = data['alpha']
        self.theta_r = data['theta_r']
        self.theta_c = data['theta_c']
        self.mg_id = data['id']
        self.residual_max = data['residual_max']
        self.eta_r_Non = None
        self.eta_c_Non = None
        self.trade = trade

        s_list = range(S)
        t_list = range(HL)
        i_list = range(I)
        M = 1e4

        model = gp.Model()
        model.setParam("OutputFlag", 1)
        model.setParam("Threads", 3)
        model.setParam("TimeLimit", 60)
        model.setParam("MIPGap", 0.1)
        model.setParam("NumericFocus", 3)  # Prioritize numerical stability
        model.setParam("FeasibilityTol", 1e-4)  # Increase feasibility tolerance
        model.setParam("Cuts", 2)

        self.g_pv = model.addVars(S, HL, lb=0, name='g_pv')  # generation by pv devices
        self.g_dg = model.addVars(S, HL, lb=0, name='g_dg')  # generation by dg devices
        self.x = model.addVars(S, HL, lb=0, name='x')   # load served
        self.r_c = model.addVars(S, HL, lb=0, name='r_c')  # charging amount
        self.r_d = model.addVars(S, HL, lb=0, name='r_d')  # discharging amount
        self.l = model.addVars(S, HL, lb=0, name='l')    # energy level at es
        self.ls = model.addVars(S, HL, lb=0, name='ls')  # load shed
        # trading variables
        self.e_sell = model.addVars(I, HL, lb=0, name='e_sell')  # energy traded
        self.e_buy = model.addVars(I, HL, lb=0, name='e_buy')    # energy traded
        self.pi_sell = model.addVars(I, HL, lb=0, name='pi_sell')  # energy traded
        self.pi_buy = model.addVars(I, HL, lb=0, name='pi_buy')    # energy traded
        self.u = model.addVars(HL, ub=1, vtype=GRB.BINARY if MIP else GRB.CONTINUOUS, name='u')
        # equity metrics
        self.eta_r = model.addVar(ub=1, name='eta_r')
        self.eta_c = model.addVar(ub=1, name='eta_c')
        # costs
        self.C_es = model.addVar(lb=0, name='C_es')  # es operation cost
        self.C_ls = model.addVar(lb=0, name='C_ls')  # load HLedding cost
        self.C_dg = model.addVar(lb=0, name='C_dg')  # dg operation cost
        self.C_u = model.addVar(lb=0, name='C_u')    # utility cost
        self.C_e = model.addVar(lb=-float('inf'), name='C_e')  # trade payments
        self.C_t = model.addVar(lb=-float('inf'), name='C_t')  # total cost

        # constraints
        model.addConstrs(
            (self.g_pv[(s, t)] <= data['pv'] * data['pv_s'][s, t] for s in s_list for t in t_list),
            name=f'g_pv_limit')
        model.addConstrs(
            (self.g_dg[(s, t)] <= data['dg'] * data['dg_ef'] for s in s_list for t in t_list),
            name=f'g_dg_limit')
        model.addConstrs(
            (self.x[(s, t)] <= data['l_s'][s, t] for s in s_list for t in t_list),
            name=f'x_limit')
        model.addConstrs(
            (self.r_c[(s, t)] <= data['doc'] * data['es'] for s in s_list for t in t_list),
            name=f'es_doc')
        model.addConstrs(
            (self.r_d[(s, t)] <= data['dod'] * data['es'] for s in s_list for t in t_list),
            name=f'es_dod')
        model.addConstrs(
            (self.l[(s, t)] <= data['es'] for s in s_list for t in t_list),
            name=f'l limit')
        model.addConstrs(
            (self.ls[(s, t)] == data['l_s'][s, t] - self.x[(s, t)] for s in s_list for t in t_list),
            name='load shed')
        model.addConstrs(
            (self.x[(s, t)] + self.r_c[(s, t)] + sum(self.e_sell[(j, t)] for j in i_list) ==
             self.g_pv[(s, t)] + self.g_dg[(s, t)] + self.r_d[(s, t)] + sum(self.e_buy[(j, t)] for j in i_list)
             for s in s_list for t in t_list),
            name=f'balance')
        model.addConstrs((self.l[(s, 0)] == data['es'] for s in s_list), name=f'l0')
        model.addConstrs(
            (self.l[(s, t+1)] == self.l[(s, t)] + self.r_c[(s, t)] * data['es_c'] -
             self.r_d[(s, t)] / data['es_d'] for s in s_list for t in t_list if t != HL-1),
            name=f'flow')
        # trade constraints
        model.addConstrs((self.e_sell[(id, t)] + self.e_buy[(id, t)] == 0 for t in t_list), name=f'self')
        model.addConstrs(
            (self.e_buy[(j, t)] <= self.u[t] * M for j in i_list for t in t_list), name='buy mode')
        model.addConstrs(
            (self.e_sell[(j, t)] <= (1 - self.u[t]) * M for j in i_list for t in t_list), name='sell mode')
        model.addConstrs(
            (self.pi_buy[(j, t)] <= data['pay_max'] * self.e_buy[(j, t)] for j in i_list for t in t_list),
            name='buy pay max')
        model.addConstrs(
            (-self.pi_buy[(j, t)] <= -data['pay_min'] * self.e_buy[(j, t)] for j in i_list for t in t_list),
            name='buy pay min')
        model.addConstrs(
            (self.pi_sell[(j, t)] <= data['pay_max'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='sell pay max')
        model.addConstrs(
            (-self.pi_sell[(j, t)] <= -data['pay_min'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='sell pay min')

        # costs
        model.addConstr(self.C_es ==
                        data['es_cost'] * sum(data['probs'][s] * sum(self.r_c[(s, t)] + self.r_d[(s, t)] for t in t_list)
                                                             for s in s_list),
                        name='C_es')
        model.addConstr(self.C_dg == data['dg_cost'] * sum(data['probs'][s] * sum(self.g_dg[(s, t)] for t in t_list)
                                                                          for s in s_list),
                        name='C_dg')
        model.addConstr(self.C_ls == data['lsp'] * sum(data['probs'][s] * sum(self.ls[(s, t)] for t in t_list)
                                                       for s in s_list),
                        name='C_ls')
        model.addConstr(self.C_u == (1 - usrr) * data['u_fee'] * (quicksum(self.e_buy) + quicksum(self.e_sell)),
                        name='C_u')
        model.addConstr(self.C_e == (1 - fsrr) * quicksum(self.pi_buy) - quicksum(self.pi_sell),
                        name='C_e')
        model.addConstr(self.C_t == self.C_es + self.C_dg + self.C_ls + self.C_u + self.C_e, name='C_t')

        # energy equity
        model.addConstr(self.eta_r == sum(data['probs'][s] * sum(self.x[(s, t)]/data['l_s'][s, t] for t in t_list)
                                          for s in s_list) / HL, name='eta_r')
        model.addConstr(self.eta_c == 1 - (self.C_t - C_t_min) / (C_t_max - C_t_min), name='eta_c')

        # Necessary: do not remove it
        model.update()
        self.model = model

        # objective function
        self.obj_equity_part = -(self.alpha * (self.theta_r * self.eta_r +
                                               self.theta_c * self.eta_c))
        self.obj_lagrangian_part = 0
        self.set_objective()
        self.update_obj()
        self.model.update()

    def set_objective(self):
        if self.trade:
            trade_off_constr = self.model.addConstr(quicksum(self.e_buy) + quicksum(self.e_sell) == 0, name='trade_off')

            self.update_obj()
            self.model.optimize()
            self.eta_r_Non = self.eta_r.x
            self.eta_c_Non = self.eta_c.x

            self.model.remove(trade_off_constr)

            self.model.addConstr(-self.eta_r <= -self.eta_r_Non, name='res_improve')
            self.model.addConstr(-self.eta_c <= -self.eta_c_Non, name='cost_improve')

            self.model.update()
        else:
            self.model.addConstr(quicksum(self.e_buy) + quicksum(self.e_sell) == 0, name='trade_off')


    def update_obj(self):
        if self.trade:
            self.model.setObjective(self.obj_equity_part + self.obj_lagrangian_part, sense=GRB.MINIMIZE)
            self.model.update()
        else:
            self.model.setObjective(self.obj_equity_part, sense=GRB.MINIMIZE)
            self.model.update()


    def get_y_values(self):
        e_buy = np.array(self.model.getAttr('X', self.e_buy.values())).reshape((self.I, self.HL))
        e_sell = np.array(self.model.getAttr('X', self.e_sell.values())).reshape((self.I, self.HL))
        pi_buy = np.array(self.model.getAttr('X', self.pi_buy.values())).reshape((self.I, self.HL))
        pi_sell = np.array(self.model.getAttr('X', self.pi_sell.values())).reshape((self.I, self.HL))

        y = np.stack((e_buy,
                      e_sell,
                      pi_buy,
                      pi_sell), axis=0)
        return y


    def get_trade_off_obj(self):
        if not self.trade:
            self.model.optimize()
            return self.eta_r.x, self.eta_c.x
        else:
            ValueError('Model is built for trading.')

    def solve_with(self, l, z_old, rho, xi=1):
        y = np.stack((np.array(self.e_buy.values()).reshape(self.I, self.HL),
                      np.array(self.e_sell.values()).reshape(self.I, self.HL),
                      np.array(self.pi_buy.values()).reshape(self.I, self.HL),
                      np.array(self.pi_sell.values()).reshape(self.I, self.HL)), axis=0)

        lagrangian_y = 0.5 * rho * np.sum((z_old - y) ** 2) + np.sum(l * (z_old - y))

        self.obj_lagrangian_part = xi * lagrangian_y

        self.model.setObjective(
            self.obj_equity_part +
            int(self.trade) * self.obj_lagrangian_part, sense=GRB.MINIMIZE)

        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")

        y = None
        if self.model.Status == GRB.OPTIMAL:
            y = self.get_y_values()
        else:
            print('rho:', rho, '\n', 'lambda:', l, '\n', 'z_old:', z_old)
            raise ValueError(f"Microgrid {self.mg_id} did not find an optimal solution."
                             f"Status: {status[self.model.Status]}")

        return y

    def save_all_solutions(self, final_l, final_z, final_rho):
        _ = self.solve_with(final_l, final_z, final_rho, xi=1)

        g_pv = np.array(self.model.getAttr('X', self.g_pv.values())).reshape((self.S, self.HL))
        g_dg = np.array(self.model.getAttr('X', self.g_dg.values())).reshape((self.S, self.HL))
        x = np.array(self.model.getAttr('X', self.x.values())).reshape((self.S, self.HL))
        r_c = np.array(self.model.getAttr('X', self.r_c.values())).reshape((self.S, self.HL))
        r_d = np.array(self.model.getAttr('X', self.r_d.values())).reshape((self.S, self.HL))
        l = np.array(self.model.getAttr('X', self.l.values())).reshape((self.S, self.HL))
        ls = np.array(self.model.getAttr('X', self.ls.values())).reshape((self.S, self.HL))
        # trading variables
        e_sell = np.array(self.model.getAttr('X', self.e_sell.values())).reshape((self.I, self.HL))
        e_buy = np.array(self.model.getAttr('X', self.e_buy.values())).reshape((self.I, self.HL))
        pi_sell = np.array(self.model.getAttr('X', self.pi_sell.values())).reshape((self.I, self.HL))
        pi_buy = np.array(self.model.getAttr('X', self.pi_buy.values())).reshape((self.I, self.HL))
        u = np.array(self.model.getAttr('X', self.u.values()))
        # equity metrics
        eta_r = self.eta_r.x
        eta_c = self.eta_c.x
        eta_r_Non = self.eta_r_Non
        eta_c_Non = self.eta_c_Non
        # costs
        C_es = self.C_es.x
        C_ls = self.C_ls.x
        C_dg = self.C_dg.x
        C_u = self.C_u.x
        C_e = self.C_e.x
        C_t = self.C_t.x

        with open(f'Solutions/microgrid{self.mg_id}.pkl', 'wb') as handle:
            pickle.dump([g_pv, g_dg, x, r_c, r_d, l, ls,
                         e_sell, e_buy, pi_sell, pi_buy, u,
                         eta_r, eta_c, eta_r_Non, eta_c_Non,
                         C_es, C_ls, C_dg, C_u, C_e, C_t], handle)
        handle.close()




