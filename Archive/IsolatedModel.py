import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import copy


class IsolatedModel:
    def __init__(self, **kwargs):
        I = kwargs['mgs_count']
        mg_id = kwargs['mg_id']
        SH = kwargs['scheduling_horizon']
        data_mg = kwargs['data_mg']
        task = kwargs['cost_task']
        S = 1
        M = 100000

        i_list = range(I)
        t_list = range(SH)
        s_list = range(S)

        self.model = gp.Model(f'SingMG')
        self.model.setParam('OutputFlag', 0)
        self.g_pv = self.model.addVars(SH, S, lb=0)
        self.g_dg = self.model.addVars(SH, S, lb=0)
        self.x = self.model.addVars(SH, S, lb=0)
        self.r_c = self.model.addVars(SH, S, lb=0)
        self.r_d = self.model.addVars(SH, S, lb=0)
        self.l = self.model.addVars(SH, S, lb=0)
        self.sh = self.model.addVars(SH, S, lb=0)
        self.e_sell = self.model.addVars(I, SH, lb=0)
        self.e_buy = self.model.addVars(I, SH, lb=0)
        self.pi_sell = self.model.addVars(I, SH, lb=0)
        self.pi_buy = self.model.addVars(I, SH, lb=0)
        self.u = self.model.addVars(SH, vtype=GRB.BINARY, name='u')
        self.C_sh = self.model.addVar(name='C_sh')
        self.C_dg = self.model.addVar(name='C_dg')
        self.C_es = self.model.addVar(name='C_es')
        self.C_u = self.model.addVar(name='C_u')  # utility cost for microgrid i
        self.C_t = self.model.addVar(lb=-float('inf'), name='C_t')
        self.C_e = self.model.addVar(lb=-float('inf'), name='C_e')
        for t in t_list:
            for s in s_list:
                ts = (t, s)
                self.model.addConstr(self.g_pv[ts] <= data_mg['pv'] * data_mg['pv_s'][s][t], name=f'g_pv[{ts}]')
                self.model.addConstr(self.g_dg[ts] <= data_mg['dg'] * data_mg['dg_ef'], name=f'g_dg[{ts}]')
                self.model.addConstr(self.x[ts] <= data_mg['l_s'][s][t], name=f'x[{ts}]')
                self.model.addConstr(self.r_c[ts] <= data_mg['doc'] * data_mg['es'], name=f'r_cd[{ts}]')
                self.model.addConstr(self.r_d[ts] <= data_mg['dod'] * data_mg['es'], name=f'r_cd[{ts}]')
                self.model.addConstr(self.l[ts] <= data_mg['es'], name=f'l[{ts}]')
                self.model.addConstr(self.sh[ts] == data_mg['l_s'][s][t] - self.x[ts], name='load shed')
                if t != SH - 1:
                    self.model.addConstr(self.l[(t + 1, s)] == self.l[(t, s)] +
                                         self.r_c[(t, s)] * data_mg['es_c'] -
                                         self.r_d[(t, s)] / data_mg['es_d'], name='flow')

        self.model.addConstrs(
            (self.l[(0, s)] == data_mg['es'] for s in s_list), name='initial l')
        self.model.addConstrs(
            (self.l[(SH-1, s)] == data_mg['es'] for s in s_list), name='last l')
        self.model.addConstrs(
            (self.x[(t, s)] + self.r_c[(t, s)] + sum(self.e_sell[(j, t)] for j in i_list) ==
            self.g_pv[(t, s)] + self.g_dg[(t, s)] + self.r_d[(t, s)] +
            sum(self.e_buy[(j, t)] for j in i_list) for s in s_list for t in t_list),
            name=f'balance')
        
        
        # trade constraints
        self.model.addConstr(sum(self.e_buy[(mg_id, t)] + self.e_sell[(mg_id, t)] for t in t_list) == 0, name='self')
        for j in i_list:
            # turn of simultaneous buy and sell
            self.model.addConstrs(
                (self.e_buy[(j, t)] <= self.u[t] * M for t in t_list),
                name='buy mode')
            self.model.addConstrs(
                (self.e_sell[(j, t)] <= (1 - self.u[t]) * M for t in t_list),
                name='sell mode')

            if task == 'min':
                self.model.addConstrs(
                    (self.pi_buy[(j, t)] == 0 * self.e_buy[(j, t)] for t in t_list),
                    name='buy pay min')
                self.model.addConstrs(
                    (self.pi_sell[(j, t)] == data_mg['pay_max'] * self.e_sell[(j, t)] for t in t_list),
                    name='sell pay max')
            elif task == 'max':
                self.model.addConstrs(
                    (self.pi_buy[(j, t)] == data_mg['pay_max'] * self.e_buy[(j, t)] for t in t_list),
                    name='buy pay min')
                self.model.addConstrs(
                    (self.pi_sell[(j, t)] == 0 * self.e_sell[(j, t)] for t in t_list),
                    name='sell pay max')
        if task == 'no trade':
            self.model.addConstr(quicksum(self.e_sell) + quicksum(self.e_buy) == 0)
            self.model.addConstr(quicksum(self.pi_sell) + quicksum(self.pi_buy) == 0)
        # costs
        self.model.addConstr(
            self.C_es == data_mg['es_cost'] * sum([data_mg['probs'][s] *
                                                          sum([self.r_c[(t, s)] + self.r_d[(t, s)]
                                                           for t in t_list])
                                                         for s in s_list]), name='C_es')
        self.model.addConstr(
            self.C_dg == data_mg['dg_cost'] * sum(data_mg['probs'][s] * sum([self.g_dg[(t, s)]
                                                                                        for t in t_list])
                                                         for s in s_list), name='C_dg')
        self.model.addConstr(
            self.C_sh == data_mg['lsp'] * sum(data_mg['probs'][s] * sum(self.sh[(t, s)] for t in t_list)
                                                     for s in s_list), name='C_sh')
        self.model.addConstr(
            self.C_u == data_mg['u_fee'] * sum(self.e_buy[(j, t)] + self.e_sell[(j, t)]
                                                       for j in i_list for t in t_list), name='C_u')
        self.model.addConstr(
            self.C_e == sum(self.pi_buy[(j, t)] - self.pi_sell[(j, t)] for j in i_list for t in t_list), name='C_e')

        self.model.addConstr(
            self.C_t == self.C_es + self.C_dg + self.C_sh + self.C_u + self.C_e, name='C_t')

        self.model.setObjective(self.C_t, sense=GRB.MINIMIZE)

        self.model.update()


