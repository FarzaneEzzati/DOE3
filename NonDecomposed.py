import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import copy


class NonDecomposed:
    def __init__(self, **kwargs):
        I = kwargs['mgs_count']
        SH = kwargs['scheduling_horizon']
        r_dis_points = kwargs['r_dis_points']
        c_dis_points = kwargs['c_dis_points']
        data = kwargs['data']
        trade = kwargs['trade']
        S = 1
        M = 100000
        if trade and r_dis_points is None and c_dis_points is None:
            raise ValueError('The model with trade must have resilience disagreement points.')
        self.Ltotal = {i: sum(data[i]['probs'][s] * sum(data[i]['l_s'][s][t] for t in range(SH)) for s in range(S))
                       for i in range(I)}
        self.model = gp.Model(f'WholeMG')
        self.model.setParam('OutputFlag', 1)
        self.model.setParam('NonConvex', 2)

        pi_min, pi_max = 1, 2
        its_list = list(product(range(I), range(SH), range(S)))
        ijt_list = list(product(range(I), range(I), range(SH)))
        ijt_list = [(i, j, t) for i, j, t in ijt_list if i != j]
        it_list = list(product(range(I), range(SH)))

        self.g_pv = self.model.addVars(its_list, lb=0)
        self.g_dg = self.model.addVars(its_list, lb=0)
        self.x = self.model.addVars(its_list, lb=0)
        self.r_c = self.model.addVars(its_list, lb=0)
        self.r_d = self.model.addVars(its_list, lb=0)
        self.l = self.model.addVars(its_list, lb=0)
        self.eta_r = self.model.addVars(range(I), ub=1, lb=0, name='eta_r')
        self.eta_c = self.model.addVars(range(I), lb=0, ub=1, name='eta_c')
        self.benefit = self.model.addVars(range(I), lb=-float('inf'), name='x_ln')

        self.ls = self.model.addVars(it_list, name='ls')
        self.C_sh = self.model.addVars(range(I), name='C_sh')
        self.C_dg = self.model.addVars(range(I), name='C_dg')
        self.C_es = self.model.addVars(range(I), name='C_es')
        self.C_t = self.model.addVars(range(I), lb=-float('inf'), name='C_t')
        self.C_t_min = {i: -SH * (data[i]['pay_max']) * (data[i]['es'] + data[i]['pv'] + data[i]['dg']) for i in range(I)}
        self.C_t_max = {i: SH * data[i]['pay_max'] * (data[i]['es'] + data[i]['l_max']) + data[i]['dg_cost'] * data[i]['dg'] +
                           data[i]['es_cost'] * data[i]['es'] + data[i]['lsp'] * data[i]['l_max']
                        for i in range(I)}

        if trade:
            # indices for z to turn off simultaneous buy and sell
            self.e_sell = self.model.addVars(ijt_list, lb=0)
            self.e_buy = self.model.addVars(ijt_list, lb=0)
            self.pi_sell = self.model.addVars(ijt_list, lb=0)
            self.pi_buy = self.model.addVars(ijt_list, lb=0)
            self.u = self.model.addVars(it_list, vtype=GRB.BINARY, name='u')
            self.C_u = self.model.addVars(range(I), lb=0, name='C_u')  # utility cost for microgrid i
            self.C_e = self.model.addVars(range(I), lb=-float('inf'), name='C_e')
            self.e_U = {(i, t): data[i]['es'] + data[i]['l_max'] for (i, t) in it_list}
            self.e_L = {(i, t): data[i]['es'] + data[i]['pv'] + data[i]['dg'] for (i, t) in it_list}

        for i in range(I):
            # list of j indices for only microgrid i
            i_trade_indice = list(range(i))+list(range(i+1, I))
            # costs
            self.model.addConstr(self.C_sh[i] ==
                                 data[i]['lsp'] * sum([data[i]['probs'][s] * sum([data[i]['l_s'][s][t] - self.x[(i, t, s)]
                                                                           for t in range(SH)])
                                                   for s in range(S)]), name='shedding cost')
            self.model.addConstr(self.C_dg[i] ==
                                 data[i]['dg_cost'] * sum([data[i]['probs'][s] * sum([self.g_dg[(i, t, s)] for t in range(SH)])
                                                     for s in range(S)]), name='fuel cost')
            self.model.addConstr(self.C_es[i] ==
                                 data[i]['es_cost'] * sum([data[i]['probs'][s] * sum([self.r_c[(i, t, s)] +
                                                                                     self.r_d[(i, t, s)]
                                                                                     for t in range(SH)])
                                                             for s in range(S)]), name='es cost')
            self.model.addConstrs((self.ls[(i, t)] == sum(data[i]['probs'][s] * (data[i]['l_s'][s][t] - self.x[(i, t, s)])
                                                          for s in range(S)) for t in range(SH)), name='load shed')

            for t in range(SH):
                for s in range(S):
                    its = (i, t, s)
                    self.model.addConstr(self.g_pv[its] <= data[i]['pv'] * data[i]['pv_s'][s][t], name=f'g_pv[{its}]')
                    self.model.addConstr(self.g_dg[its] <= data[i]['dg'], name=f'g_dg[{its}]')
                    self.model.addConstr(self.x[its] <= data[i]['l_s'][s][t], name=f'x[{its}]')
                    self.model.addConstr(self.r_c[its] <= data[i]['doc'] * data[i]['es'], name=f'r_cd[{its}]')
                    self.model.addConstr(self.r_d[its] <= data[i]['dod'] * data[i]['es'], name=f'r_cd[{its}]')
                    self.model.addConstr(self.l[its] <= data[i]['es'], name=f'l[{its}]')
                    if t != SH - 1:
                        self.model.addConstr(self.l[(i, t + 1, s)] == self.l[(i, t, s)] +
                                             self.r_c[(i, t, s)] - self.r_d[(i, t, s)], name='flow')

            self.model.addConstrs(
                (self.l[(i, 0, s)] == data[i]['es'] for s in range(S)), name='initial l')
            self.model.addConstr(
                self.eta_r[i] == sum([data[i]['probs'][s] * sum([self.x[(i, t, s)] / (0.0001 + data[i]['l_s'][s][t])
                                                              for t in range(SH)]) / SH for s in range(S)]),
                name='resilience')
            self.model.addConstr(
                self.eta_c[i] == 1 - (self.C_t[i] - self.C_t_min[i]) / (self.C_t_max[i] - self.C_t_min[i]),
                name='eta_c')
            self.model.addConstr(
                self.benefit[i] == data[i]['alpha'] * (0.85 * (self.eta_r[i] - r_dis_points[i]) +
                                                       0.15 * (self.eta_c[i] - c_dis_points[i])), name='benefit')
            if trade:
                self.model.addConstrs(
                    (self.x[(i, t, s)] + self.r_c[(i, t, s)] + sum(self.e_sell[(i, j, t)] for j in i_trade_indice) ==
                    self.g_pv[(i, t, s)] + self.g_dg[(i, t, s)] + self.r_d[(i, t, s)] +
                    sum(self.e_buy[(i, j, t)] for j in i_trade_indice) for s in range(S) for t in range(SH)),
                    name=f'balance')
                for j in i_trade_indice:
                    self.model.addConstrs(
                        (self.e_buy[(i, j, t)] - self.e_sell[(j, i, t)] == 0 for t in range(SH)), name='trade clear')
                    self.model.addConstrs(
                        (self.pi_buy[(i, j, t)] - self.pi_sell[(j, i, t)] == 0 for t in range(SH)), name='pay clear')
                    # power quality
                    self.model.addConstrs(
                        (self.e_buy[(i, j, t)] <= self.u[(i, t)] * M for t in range(SH)),
                        name='power quality buy')
                    self.model.addConstrs(
                        (self.e_sell[(i, j, t)] <= (1 - self.u[(i, t)]) * M for t in range(SH)),
                        name='power quality sell')

                    # turn of simultaneous buy and sell
                    self.model.addConstrs(
                        (self.e_buy[(i, j, t)] <= M * self.u[(i, t)] for t in range(SH)),
                        name='e buy')
                    self.model.addConstrs(
                        (self.e_sell[(i, j, t)] <= M * (1 - self.u[(i, t)]) for t in range(SH)),
                        name='e sell')

                    self.model.addConstrs(
                        (self.pi_sell[(i, j, t)] <= data[i]['pay_max'] * self.e_sell[(i, j, t)] for t in range(SH)),
                        name='pi_sell max')
                    self.model.addConstrs(
                        (-self.pi_sell[(i, j, t)] <= -data[i]['pay_min'] * self.e_sell[(i, j, t)] for t in range(SH)),
                        name='pi_sell_min')
                    self.model.addConstrs(
                        (self.pi_buy[(i, j, t)] <= data[i]['pay_max'] * self.e_buy[(i, j, t)] for t in range(SH)),
                        name='pi_buy_max')
                    self.model.addConstrs(
                        (-self.pi_buy[(i, j, t)] <= -data[i]['pay_min'] * self.e_buy[(i, j, t)] for t in range(SH)),
                        name='pi_buy_min')

                self.model.addConstr(
                    self.C_e[i] == sum(self.pi_buy[(i, j, t)] - self.pi_sell[(i, j, t)]
                                       for j in i_trade_indice for t in range(SH)), name='C_e')
                self.model.addConstr(
                    self.C_u[i] == data[i]['u_fee'] * sum(self.e_buy[(i, j, t)] + self.e_sell[(i, j, t)]
                                       for j in i_trade_indice for t in range(SH)), name='C_e')
                self.model.addConstr(
                    self.C_t[i] == self.C_sh[i] + self.C_es[i] + self.C_dg[i] + self.C_u[i] + self.C_e[i], name='C_t')
                self.model.addConstr(
                    -self.eta_r[i] <= -r_dis_points[i], name='resilience')
                self.model.addConstr(
                    -self.eta_c[i] <= -c_dis_points[i], name='cost')
            else:
                self.model.addConstr(self.C_t[i] == self.C_sh[i] + self.C_es[i] + self.C_dg[i],
                                     name='C_t')
                self.model.addConstrs((self.x[(i, t, s)] + self.r_c[(i, t, s)] ==
                                       self.g_pv[(i, t, s)] + self.g_dg[(i, t, s)] + self.r_d[(i, t, s)]
                                       for s in range(S) for t in range(SH)), name=f'balance')
        self.model.setObjective(quicksum(self.benefit), sense=GRB.MAXIMIZE)

        self.model.update()


