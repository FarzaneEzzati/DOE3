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
        data_mgs = kwargs['data_mgs']
        data_subsidies = kwargs['data_subsidies']
        trade = kwargs['trade']
        thetas = kwargs['thetas']
        S = 1
        M = 100000

        i_list = range(I)
        t_list = range(SH)
        s_list = range(S)
        fsrr = {i: data_subsidies['fsr'] * data_mgs[i]['sv'] for i in i_list}
        usrr = {i: data_subsidies['usr'] * data_mgs[i]['sv'] for i in i_list}

        if trade and r_dis_points is None and c_dis_points is None:
            raise ValueError('The model with trade must have resilience disagreement points.')

        self.model = gp.Model(f'WholeMG')
        self.model.setParam('OutputFlag', 0)
        self.g_pv = self.model.addVars(I, SH, S, lb=0)
        self.g_dg = self.model.addVars(I, SH, S, lb=0)
        self.x = self.model.addVars(I, SH, S, lb=0)
        self.r_c = self.model.addVars(I, SH, S, lb=0)
        self.r_d = self.model.addVars(I, SH, S, lb=0)
        self.l = self.model.addVars(I, SH, S, lb=0)
        self.eta_r = self.model.addVars(I, lb=0, ub=1, name='eta_r')
        self.eta_c = self.model.addVars(I, lb=0, ub=1, name='eta_c')
        self.sh = self.model.addVars(I, SH, S, lb=0)
        self.e_sell = self.model.addVars(I, I, SH, lb=0)
        self.e_buy = self.model.addVars(I, I, SH, lb=0)
        self.pi_sell = self.model.addVars(I, I, SH, lb=0)
        self.pi_buy = self.model.addVars(I, I, SH, lb=0)
        self.u = self.model.addVars(I, SH, vtype=GRB.BINARY, name='u')
        self.C_sh = self.model.addVars(I, name='C_sh')
        self.C_dg = self.model.addVars(I, name='C_dg')
        self.C_es = self.model.addVars(I, name='C_es')
        self.C_u = self.model.addVars(I, name='C_u')  # utility cost for microgrid i
        self.C_e = self.model.addVars(I, lb=-float('inf'), name='C_e')
        self.C_t = self.model.addVars(I, lb=-float('inf'), name='C_t')

        self.C_t_min = {i: data_mgs[i]['C_t_min'] for i in i_list}
        self.C_t_max = {i: data_mgs[i]['C_t_max'] for i in i_list}
        #self.C_t_min = {i: -10000 for i in i_list}
        #self.C_t_max = {i: 20000 for i in i_list}

        for i in i_list:
            for t in t_list:
                for s in s_list:
                    its = (i, t, s)
                    self.model.addConstr(self.g_pv[its] <= data_mgs[i]['pv'] * data_mgs[i]['pv_s'][s][t], name=f'g_pv[{its}]')
                    self.model.addConstr(self.g_dg[its] <= data_mgs[i]['dg'] * data_mgs[i]['dg_ef'], name=f'g_dg[{its}]')
                    self.model.addConstr(self.x[its] <= data_mgs[i]['l_s'][s][t], name=f'x[{its}]')
                    self.model.addConstr(self.r_c[its] <= data_mgs[i]['doc'] * data_mgs[i]['es'], name=f'r_cd[{its}]')
                    self.model.addConstr(self.r_d[its] <= data_mgs[i]['dod'] * data_mgs[i]['es'], name=f'r_cd[{its}]')
                    self.model.addConstr(self.l[its] <= data_mgs[i]['es'], name=f'l[{its}]')
                    self.model.addConstr(self.sh[its] == data_mgs[i]['l_s'][s][t] - self.x[its], name='load shed')
                    if t != SH - 1:
                        self.model.addConstr(self.l[(i, t + 1, s)] == self.l[(i, t, s)] +
                                             self.r_c[(i, t, s)] * data_mgs[i]['es_c'] - 
                                             self.r_d[(i, t, s)] / data_mgs[i]['es_d'], name='flow')

            self.model.addConstrs(
                (self.l[(i, 0, s)] == data_mgs[i]['es'] for s in s_list), name='initial l')
            self.model.addConstrs(
                (self.l[(i, SH-1, s)] == data_mgs[i]['es'] for s in s_list), name='last l')
            self.model.addConstrs(
                (self.x[(i, t, s)] + self.r_c[(i, t, s)] + sum(self.e_sell[(i, j, t)] for j in i_list) ==
                self.g_pv[(i, t, s)] + self.g_dg[(i, t, s)] + self.r_d[(i, t, s)] +
                sum(self.e_buy[(i, j, t)] for j in i_list) for s in s_list for t in t_list),
                name=f'balance')

            # trade constraints
            self.model.addConstr(sum(self.e_buy[(i, i, t)] + self.e_sell[(i, i, t)] for t in t_list) == 0, name='self')
            for j in i_list:
                # clearance
                self.model.addConstrs(
                    (self.e_buy[(i, j, t)] - self.e_sell[(j, i, t)] == 0 for t in t_list), name='trade clear')
                self.model.addConstrs(
                    (self.pi_buy[(i, j, t)] - self.pi_sell[(j, i, t)] == 0 for t in t_list), name='pay clear')
                # turn of simultaneous buy and sell
                self.model.addConstrs(
                    (self.e_buy[(i, j, t)] <= self.u[(i, t)] * M for t in t_list),
                    name='buy mode')
                self.model.addConstrs(
                    (self.e_sell[(i, j, t)] <= (1 - self.u[(i, t)]) * M for t in t_list),
                    name='sell mode')
                
                self.model.addConstrs(
                    (self.pi_sell[(i, j, t)] <= data_mgs[i]['pay_max'] * self.e_sell[(i, j, t)] for t in t_list),
                    name='sell pay max')
                self.model.addConstrs(
                    (-self.pi_sell[(i, j, t)] <= -data_mgs[i]['pay_min'] * self.e_sell[(i, j, t)] for t in t_list),
                    name='sell pay min')
                self.model.addConstrs(
                    (self.pi_buy[(i, j, t)] <= data_mgs[i]['pay_max'] * self.e_buy[(i, j, t)] for t in t_list),
                    name='buy pay max')
                self.model.addConstrs(
                    (-self.pi_buy[(i, j, t)] <= -data_mgs[i]['pay_min'] * self.e_buy[(i, j, t)] for t in t_list),
                    name='buy pay min')

            # costs
            self.model.addConstr(
                self.C_es[i] == data_mgs[i]['es_cost'] * sum([data_mgs[i]['probs'][s] *
                                                              sum([self.r_c[(i, t, s)] + self.r_d[(i, t, s)]
                                                               for t in t_list])
                                                             for s in s_list]), name='C_es')
            self.model.addConstr(
                self.C_dg[i] == data_mgs[i]['dg_cost'] * sum(data_mgs[i]['probs'][s] * sum([self.g_dg[(i, t, s)]
                                                                                            for t in t_list])
                                                             for s in s_list), name='C_dg')
            self.model.addConstr(
                self.C_sh[i] == data_mgs[i]['lsp'] * sum(data_mgs[i]['probs'][s] * sum(self.sh[(i, t, s)] for t in t_list)
                                                         for s in s_list), name='C_sh')
            self.model.addConstr(
                self.C_u[i] ==
                (1 - usrr[i]) * data_mgs[i]['u_fee'] * sum(self.e_buy[(i, j, t)] + self.e_sell[(i, j, t)]
                                                           for j in i_list for t in t_list), name='C_u')
            self.model.addConstr(
                self.C_e[i] ==
                (1 - fsrr[i]) * sum(self.pi_buy[(i, j, t)] for j in i_list for t in t_list) -
                sum(self.pi_sell[(i, j, t)] for j in i_list for t in t_list), name='C_e')

            self.model.addConstr(
                self.C_t[i] == self.C_es[i] + self.C_dg[i] + self.C_sh[i] + self.C_u[i] + self.C_e[i], name='C_t')

            # equity metric
            self.model.addConstr(
                self.eta_r[i] == sum(data_mgs[i]['probs'][s] * sum(self.x[(i, t, s)]/data_mgs[i]['l_s'][s][t]
                                                                 for t in t_list) for s in s_list) / SH,
                name='eta_r')
            self.model.addConstr(
                self.eta_c[i] == 1 - (self.C_t[i] - self.C_t_min[i]) / (self.C_t_max[i] - self.C_t_min[i]),
                name='eta_c')
            self.model.addConstr(-self.eta_r[i] <= -r_dis_points[i], name='eta_r improve')
            self.model.addConstr(-self.eta_c[i] <= -c_dis_points[i], name='eta_c improve')

        self.model.addConstr(
            sum(fsrr[i] * sum(self.pi_buy[(i, j, t)] for j in i_list for t in t_list) for i in i_list) <=
            data_subsidies['FSS'], name='financial subsidy cap')
        self.model.addConstr(
            sum(usrr[i] * sum(self.e_buy[(i, j, t)] + self.e_sell[(i, j, t)] for j in i_list for t in t_list)
                for i in i_list) <= data_subsidies['USS'], name='utility subsidy cap')
        if not trade:
            self.model.addConstr(quicksum(self.e_buy) + quicksum(self.e_sell) == 0, name='no trade')
            self.model.addConstr(quicksum(self.pi_buy) + quicksum(self.pi_sell) == 0, name='no trade')

        self.model.setObjective(sum(data_mgs[i]['alpha'] * (thetas['r'] * (self.eta_r[i] - r_dis_points[i]) +
                                                            thetas['c'] * (self.eta_c[i] - c_dis_points[i]))
                                    for i in i_list), sense=GRB.MAXIMIZE)
        self.model.update()


