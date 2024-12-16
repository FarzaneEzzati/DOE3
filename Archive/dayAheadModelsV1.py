import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import copy

I = 4
SH = 24
eps = 1e-8  # for load scenarios with 0 value


def wholeModel(data):
    buy_min, buy_max = 0.1, 1
    sell_min, sell_max = 0.1, 1
    model = gp.Model(f'WholeMG')
    model.setParam('OutputFlag', 0)
    tij_list = list(product(range(SH), range(I), range(I)))
    # trading variables
    e = model.addVars(tij_list, lb=-float('inf'), name='e')  # energy traded +=sell, -=purchase
    pi = model.addVars(tij_list, lb=-float('inf'), name='pi')  # trade payment

    S = 1
    ts_list = list(product(range(SH), range(S)))
    g_pv = {i: model.addVars(ts_list, lb=0) for i in range(I)}
    x = {i: model.addVars(ts_list, lb=0) for i in range(I)}
    r_c = {i: model.addVars(ts_list, lb=0) for i in range(I)}
    r_d = {i: model.addVars(ts_list, lb=0) for i in range(I)}
    l = {i: model.addVars(ts_list, lb=0) for i in range(I)}
    costs = 0
    for i in range(I):
        # constraints: name can be retrieved by 'cnstr_name[(i, j)]'
        for ts in ts_list:
            t, s = ts[0], ts[1]
            model.addConstr(g_pv[i][ts] <= data[i].devices['pv'] * data[i].pv_scenarios[s][t])
            model.addConstr(x[i][ts] <= data[i].load_scenarios[s][t])
            model.addConstr(r_c[i][ts] + r_d[i][ts] <= data[i].es_depths['doc'] * data[i].devices['es'])
            model.addConstr(l[i][ts] <= data[i].devices['es'])
            model.addConstr(x[i][ts] + r_c[i][ts] ==
                            g_pv[i][ts] + r_d[i][ts] - quicksum(e[t, i, j] for j in range(I)))
            if t == 0:
                model.addConstr(l[i][ts] == data[i].devices['es'])
            elif t == SH - 1:
                model.addConstr(l[i][ts] == data[i].devices['es'])
            else:
                model.addConstr(l[i][ts] == l[i][ts] +
                                r_c[i][ts] * data[i].effic['es charge effic'] -
                                r_d[i][ts] / data[i].effic['es discharge effic'])

        # costs
        C_e = quicksum(pi)
        C_sh = sum(data[i].gamma['sh'] * sum(
            [data[i].probs[s] * sum([data[i].load_scenarios[s][t] - x[i][(t, s)] for t in range(SH)]) for s in
             range(S)]) for i in range(I))
        C_es = sum(data[i].gamma['es'] * sum(
            [data[i].probs[s] * sum([r_c[i][(t, s)] + r_d[i][(t, s)] for t in range(SH)]) for s in range(S)]) for i in
                                    range(I))
        for j in range(I):
            if i != j:
                model.addConstrs(e[(t, i, j)] + e[(t, j, i)] == 0 for t in range(SH))
                model.addConstrs(pi[(t, i, j)] + pi[(t, j, i)] == 0 for t in range(SH))
        costs += C_sh + C_es - C_e
    model.setObjective(costs, sense=GRB.MINIMIZE)
    model.update()
    return model


class LowerLevelProblem:
    def __init__(self, mg_id, data_dict, rho_dict):
        data = data_dict
        self.rho_e, self.rho_pi = rho_dict['e'], rho_dict['pi']
        self.tij_list = list(product(range(SH), [mg_id], range(I)))

        S = len(data.probs)  # count of scenarios
        st_list = list(product(range(S), range(SH)))

        # finding max sell and purchase amounts
        buy_min, buy_max = 0.1, 1
        sell_min, sell_max = 0.1, 1

        model = gp.Model(f'P1-MG{mg_id}')
        model.setParam('OutputFlag', 0)
        self.g_pv = model.addVars(st_list, lb=0, name='g_pv')  # generation by pv devices
        self.g_dg = model.addVars(st_list, lb=0, name='g_dg')  # generation by dg devices
        self.x = model.addVars(st_list, lb=0, name='x')  # load served
        self.r_c = model.addVars(st_list, lb=0, name='r_c')  # charging amount
        self.r_d = model.addVars(st_list, lb=0, name='r_d')  # discharging amount
        self.l = model.addVars(st_list, lb=0, name='l')  # energy level at es
        # trading variables
        self.e = model.addVars(self.tij_list, lb=-float('inf'), name='e')  # energy traded +=sell, -=purchase
        self.pi = model.addVars(self.tij_list, lb=-float('inf'), name='pi')  # trade payment
        self.e_sell = model.addVars(self.tij_list, name='e_sell')
        self.e_buy = model.addVars(self.tij_list, name='e_buy')
        self.pi_sell = model.addVars(self.tij_list, name='pi_sell')
        self.pi_buy = model.addVars(self.tij_list, name='pi_buy')
        # cost variables to track the optimal solution
        self.C_e = model.addVar(lb=-float('inf'), name='C_e')
        self.C_sh = model.addVar(name='C_sh')
        self.C_dg = model.addVar(name='C_dg')
        self.C_es = model.addVar(name='C_es')
        self.eta_er = model.addVar(name='eta_er')

        self.e_hat = dict.fromkeys(self.tij_list, 0)  # auxiliary trade variable, set fixed
        self.pi_hat = dict.fromkeys(self.tij_list, 0)  # auxiliary payment variable, set fixed
        self.lambda_e = dict.fromkeys(self.tij_list, 0)  # dual variable for trade, set fixed
        self.lambda_pi = dict.fromkeys(self.tij_list, 0)  # dual variable for payment, set fixed

        # constraints: name can be retrieved by 'cnstr_name[(i, j)]'
        for st in st_list:
            s, t = st[0], st[1]
            model.addConstr(self.g_pv[st] <= data.devices['pv'] * data.pv_scenarios[s][t], name=f'g_pv_limit[{st}]')
            model.addConstr(self.g_dg[st] <= data.devices['dg'] * data.effic['dg effic'], name=f'g_dg_limit[{st}]')
            model.addConstr(self.x[st] <= data.load_scenarios[s][t], name=f'x_limit[{st}]')
            model.addConstr(self.r_c[st] <= data.es_depths['doc'] * data.devices['es'], name=f'es_doc[{st}]')
            model.addConstr(self.r_d[st] <= data.es_depths['dod'] * data.devices['es'], name=f'es_dod[{st}]')
            model.addConstr(self.l[st] <= data.devices['es'], name=f'l limit[{st}]')
            model.addConstr(self.x[st] + self.r_c[st] ==
                            self.g_pv[st] + self.g_dg[st] + self.r_d[st] - quicksum(
                self.e[t, mg_id, j] for j in range(I)), name=f'balance[{st}]')
            if t == 0:
                model.addConstr(self.l[(s, t)] == data.devices['es'], name=f'l0[{st}]')
            elif t == SH - 1:
                model.addConstr(self.l[(s, t)] == data.devices['es'], name=f'lT[{st}]')
            else:
                model.addConstr(self.l[(s, t + 1)] == self.l[st] +
                                self.r_c[st] * data.effic['es charge effic'] -
                                self.r_d[st] / data.effic['es discharge effic'], name=f'es_level[{st}]')
        for t in range(SH):
            model.addConstr((self.e[t, mg_id, mg_id] == 0), name=f'self_trade[{t}]')
            model.addConstr((self.pi[t, mg_id, mg_id] == 0), name=f'self_payment[{t}]')
            model.addConstr((self.e_sell[t, mg_id, mg_id] == 0), name=f'self_e_sell[{t}]')
            model.addConstr((self.e_buy[t, mg_id, mg_id] == 0), name=f'self_e_buy[{t}]')
            model.addConstr((self.pi_sell[t, mg_id, mg_id] == 0), name=f'self_pi_sell[{t}]')
            model.addConstr((self.pi_buy[t, mg_id, mg_id] == 0), name=f'self_pi_buy[{t}]')

        model.addConstrs((self.e[tij] == self.e_sell[tij] - self.e_buy[tij] for tij in self.tij_list),
                         name='e_decomposed')
        model.addConstrs((self.pi[tij] == self.pi_sell[tij] - self.pi_buy[tij] for tij in self.tij_list),
                         name='pi_decomposed')
        model.addConstrs((self.pi_sell[tij] >= sell_min * (self.e_sell[tij]) for tij in self.tij_list),
                         name='pi_sell_min')
        model.addConstrs((self.pi_sell[tij] <= sell_max * (self.e_sell[tij]) for tij in self.tij_list),
                         name='pi_sell_max')
        model.addConstrs((self.pi_buy[tij] >= buy_min * (self.e_buy[tij]) for tij in self.tij_list),
                         name='pi_buy_min')
        model.addConstrs((self.pi_buy[tij] <= buy_max * (self.e_buy[tij]) for tij in self.tij_list),
                         name='pi_buy_max')

        # costs
        model.addConstr(self.C_e == quicksum(self.pi), name='revenue')
        model.addConstr(self.C_sh == data.gamma['sh'] * sum(
            [data.probs[s] * sum([data.load_scenarios[s][t] - self.x[(s, t)] for t in range(SH)]) for s in range(S)]),
                        name='shedding cost')
        model.addConstr(self.C_dg == data.gamma['f'] * sum(
            [data.probs[s] * sum([self.g_dg[(s, t)] for t in range(SH)]) for s in range(S)]), name='fuel cost')
        model.addConstr(self.C_es == data.gamma['es'] * sum(
            [data.probs[s] * sum([self.r_c[(s, t)] + self.r_d[(s, t)] for t in range(SH)]) for s in range(S)]),
                        name='es cost')

        # energy resilience
        model.addConstr(self.eta_er == sum(
            [data.probs[s] * sum([self.x[(s, t)] / (eps + data.load_scenarios[s][t]) for t in range(SH)]) / SH for s in
             range(S)]))

        # Necessary: do not remove it
        self.model = model

        # objective function
        # + data.thetas['trade'] * self.C_e_tilde
        self.obj_cost_part = self.C_sh + self.C_dg + self.C_es
        self.obj_revenue_part = -self.C_e
        self.obj_updating_part = 0  # just define it here, next line creates the linear expression of it
        self.buildLagrangianTerms(self.e_hat, self.pi_hat, self.lambda_e, self.lambda_pi)
        self.updateObjective()

    def buildLagrangianTerms(self, e_hat_k, pi_hat_k, lambda_e_k, lambda_pi_k):
        self.e_hat, self.pi_hat, self.lambda_e, self.lambda_pi = e_hat_k, pi_hat_k, lambda_e_k, lambda_pi_k
        Lagrangian_trade = sum(0.5 * self.rho_e * (self.e_hat[tij] - self.e[tij]) ** 2 +
                               self.lambda_e[tij] * (self.e_hat[tij] - self.e[tij]) for tij in self.tij_list)
        Lagrangian_payment = sum(0.5 * self.rho_pi * (self.pi_hat[tij] - self.pi[tij]) ** 2 +
                                 self.lambda_pi[tij] * (self.pi_hat[tij] - self.pi[tij]) for tij in self.tij_list)
        self.obj_updating_part = Lagrangian_trade + Lagrangian_payment

    def updateObjective(self):
        self.model.setObjective(self.obj_cost_part + self.obj_revenue_part + self.obj_updating_part, sense=GRB.MINIMIZE)
        self.model.update()


class HigherLevelProblem:
    def __init__(self, epsilon, rho_dict):
        self.done = False
        self.EPSILON = epsilon
        self.rho_e = rho_dict['e']
        self.rho_pi = rho_dict['pi']
        self.gaps = {'e': [], 'pi': []}

        # original variables e, pi, lambda_k
        self.e = dict.fromkeys(range(I))
        self.pi = dict.fromkeys(range(I))

        # auxiliary variables e_hat, pi_hat, lambda_k+1
        self.e_hat = dict.fromkeys(range(I))
        self.pi_hat = dict.fromkeys(range(I))
        self.lambda_e = dict.fromkeys(range(I))
        self.lambda_pi = dict.fromkeys(range(I))

        for i in range(I):
            tij_list = list(product(range(SH), [i], range(I)))
            self.e_hat[i] = dict.fromkeys(tij_list, 0)
            self.pi_hat[i] = dict.fromkeys(tij_list, 0)
            self.lambda_e[i] = dict.fromkeys(tij_list, 0)
            self.lambda_pi[i] = dict.fromkeys(tij_list, 0)

    def updateAuxiliaryVars(self, i, j):
        for t in range(SH):
            self.e_hat[i][(t, i, j)] = (1 / (2 * self.rho_e)) * (
                    self.rho_e * (self.e[i][(t, i, j)] - self.e[j][(t, j, i)]) - (
                    self.lambda_e[i][(t, i, j)] - self.lambda_e[j][(t, j, i)]))
            self.e_hat[j][(t, j, i)] = -copy.copy(self.e_hat[i][(t, i, j)])

            self.pi_hat[i][(t, i, j)] = (1 / (2 * self.rho_pi)) * (
                    self.rho_pi * (self.pi[i][(t, i, j)] - self.pi[j][(t, j, i)]) - (
                    self.lambda_pi[i][(t, i, j)] - self.lambda_pi[j][(t, j, i)]))
            self.pi_hat[j][(t, j, i)] = -copy.copy(self.pi_hat[i][(t, i, j)])

            self.lambda_e[i][(t, i, j)] = self.lambda_e[i][(t, i, j)] + self.rho_e * (
                    self.e_hat[i][(t, i, j)] - self.e[i][(t, i, j)])
            self.lambda_e[j][(t, j, i)] = self.lambda_e[j][(t, j, i)] + self.rho_e * (
                    self.e_hat[j][(t, j, i)] - self.e[j][(t, j, i)])

            self.lambda_pi[i][(t, i, j)] = self.lambda_pi[i][(t, i, j)] + self.rho_pi * (
                    self.pi_hat[i][(t, i, j)] - self.pi[i][(t, i, j)])
            self.lambda_pi[j][(t, j, i)] = self.lambda_pi[j][(t, j, i)] + self.rho_pi * (
                    self.pi_hat[j][(t, j, i)] - self.pi[j][(t, j, i)])

    def checkConvergence(self):
        e_gap = []
        pi_gap = []
        for i in range(I):
            e_gap.append(
                np.sum(np.abs(np.subtract(list(self.e[i].values()), list(self.e_hat[i].values()))) > self.EPSILON))
            pi_gap.append(
                np.sum(np.abs(np.subtract(list(self.pi[i].values()), list(self.pi_hat[i].values()))) > self.EPSILON))
        if np.sum(e_gap) == 0 and np.sum(pi_gap) == 0:
            self.done = True
        self.gaps['e'].append(np.sum(e_gap))
        self.gaps['pi'].append(np.sum(pi_gap))
