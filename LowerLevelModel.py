import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import copy


class LowerLevelModel:
    def __init__(self, **kwargs):
        mg_id = kwargs['mg_id']
        I = kwargs['mg_count']
        SH = kwargs['scheduling_horizon']
        data = kwargs['mg_data']
        thetas = kwargs['thetas']

        big_M = 10000
        S = len(data['probs'])  # count of scenarios
        s_list = range(S)
        t_list = range(SH)
        i_list = range(I)
        model = gp.Model(f'P1-MG{mg_id}')
        model.setParam('OutputFlag', 0)

        self.g_pv = model.addVars(S, SH, lb=0, name='g_pv')  # generation by pv devices
        self.g_dg = model.addVars(S, SH, lb=0, name='g_dg')  # generation by dg devices
        self.x = model.addVars(S, SH, lb=0, name='x')  # load served
        self.r_c = model.addVars(S, SH, lb=0, name='r_c')  # charging amount
        self.r_d = model.addVars(S, SH, lb=0, name='r_d') # discharging amount
        self.l = model.addVars(S, SH, lb=0, name='l')  # energy level at es
        # trading variables
        self.e_sell = model.addVars(I, SH, lb=0, name='e_sell')  # energy traded
        self.e_buy = model.addVars(I, SH, lb=0, name='e_buy')  # energy traded
        self.pi_sell = model.addVars(I, SH, lb=0, name='pi_buy')  # energy traded
        self.pi_buy = model.addVars(I, SH, lb=0, name='pi_sell')  # energy traded
        self.u = model.addVars(SH, vtype=GRB.BINARY, name='u')
        #self.u = model.addVars(SH, ub=1, name='u')
        # equity metrics
        self.eta_r = model.addVar(ub=1, name='eta_r')
        self.eta_c = model.addVar(ub=1, name='eta_c')
        # costs
        self.C_es = model.addVar(lb=0, name='C_es')  # es operation cost
        self.C_ls = model.addVar(lb=0, name='C_ls')  # load shedding cost
        self.C_dg = model.addVar(lb=0, name='C_dg')  # dg operation cost
        self.C_u = model.addVar(lb=0, name='C_u')  # utility cost
        self.C_e = model.addVar(lb=-float('inf'), name='C_e')  # trade payments
        self.C_t = model.addVar(lb=-float('inf'), name='C_t')

        # Parameters
        self.C_t_min = -SH * (data['pay_max']) * (data['es'] + data['pv'] + data['dg'])
        self.C_t_max = SH * (data['pay_max'] * (data['es'] + data['l_max']) +
                             data['dg_cost'] * data['dg'] +
                             data['es_cost'] * data['es'] +
                             data['lsp'] * data['l_max'])

        model.addConstrs(
            (self.g_pv[(s, t)] <= data['pv'] * data['pv_s'][s][t] for s in s_list for t in t_list),
            name=f'g_pv_limit')
        model.addConstrs(
            (self.g_dg[(s, t)] <= data['dg'] * data['dg_ef'] for s in s_list for t in t_list),
            name=f'g_dg_limit')
        model.addConstrs(
            (self.x[(s, t)] <= data['l_s'][s][t] for s in s_list for t in t_list),
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
            (self.x[(s, t)] + self.r_c[(s, t)] + sum(self.e_sell[(j, t)] for j in i_list) ==
             self.g_pv[(s, t)] + self.g_dg[(s, t)] + self.r_d[(s, t)] + sum(self.e_buy[(j, t)] for j in i_list)
             for s in s_list for t in t_list),
            name=f'balance')
        model.addConstrs((self.l[(s, 0)] == data['es'] for s in s_list), name=f'l0')
        model.addConstrs((self.l[(s, SH-1)] == data['es'] for s in s_list), name=f'lT')
        model.addConstrs(
            (self.l[(s, t+1)] == self.l[(s, t)] + self.r_c[(s, t)] * data['es_c'] -
             self.r_d[(s, t)] / data['es_d'] for s in s_list for t in t_list if t != SH-1),
            name=f'es_level')
        # trade constraints
        model.addConstrs((self.e_sell[(mg_id, t)] == 0 for t in t_list), name=f'self sell')
        model.addConstrs((self.e_buy[(mg_id, t)] == 0 for t in t_list), name=f'self buy')
        model.addConstrs(
            (self.e_sell[(j, t)] <= self.u[t] * big_M for j in i_list for t in t_list), name='sell mode')
        model.addConstrs(
            (self.e_buy[(j, t)] <= (1 - self.u[t]) * big_M for j in i_list for t in t_list), name='buy mode')

        model.addConstrs(
            (self.pi_sell[(j, t)] <= data['pay_max'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='sell pay max')
        model.addConstrs(
            (-self.pi_sell[(j, t)] <= -data['pay_min'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='sell pay min')
        model.addConstrs(
            (self.pi_buy[(j, t)] <= data['pay_max'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='buy pay max')
        model.addConstrs(
            (-self.pi_buy[(j, t)] <= -data['pay_min'] * self.e_sell[(j, t)] for j in i_list for t in t_list),
            name='buy pay min')

        # costs
        model.addConstr(self.C_es ==
                        data['es_cost'] * sum(data['probs'][s] * sum(self.r_c[(s, t)] + self.r_d[(s, t)] for t in t_list)
                                                             for s in s_list),
                        name='C_es')
        model.addConstr(self.C_dg == data['dg_cost'] * sum(data['probs'][s] * sum(self.g_dg[(s, t)] for t in t_list)
                                                                          for s in s_list),
                        name='C_dg')
        model.addConstr(self.C_ls == data['lsp'] * sum(data['probs'][s] * sum(data['l_s'][s][t] - self.x[(s, t)] for t in t_list)
                                                                      for s in s_list)
                        , name='C_ls')
        model.addConstr(self.C_u == data['u_fee'] * quicksum(self.e_buy) + quicksum(self.e_sell),
                        name='C_u')
        model.addConstr(self.C_e == quicksum(self.pi_buy) - quicksum(self.pi_sell),
                        name='C_e')
        model.addConstr(self.C_t == self.C_es + self.C_dg + self.C_ls + self.C_u + self.C_e, name='C_t')

        # energy equity
        model.addConstr(self.eta_r == sum(data['probs'][s] * sum(self.x[(s, t)]/(SH * data['l_s'][s][t]) for t in t_list)
                                          for s in s_list),
                        name='eta_r')
        model.addConstr(self.eta_c == 1 - (self.C_t - self.C_t_min) / (self.C_t_max - self.C_t_min), name='eta_c')

        # Necessary: do not remove it
        self.model = model

        # objective function
        self.obj_fixed_part = -(data['alpha'] * (thetas['r'] * (self.eta_r - data['eta_r_Non']) +
                                                 thetas['c'] * (self.eta_c - data['eta_c_Non'])))
        self.obj_updating_part = 0
        self.I = I
        self.SH = SH
    def buildLagrangianTerms(self, e_buy_hat, e_sell_hat, pi_buy_hat, pi_sell_hat, l_e_buy, l_e_sell, l_pi_buy, l_pi_sell, rhos):

        e_buy = np.array(self.e_buy.values()).reshape(self.I, self.SH)
        e_sell = np.array(self.e_sell.values()).reshape(self.I, self.SH)
        pi_buy = np.array(self.pi_buy.values()).reshape(self.I, self.SH)
        pi_sell = np.array(self.pi_sell.values()).reshape(self.I, self.SH)

        L_e_buy = 0.5 * rhos['e'] * ((e_buy_hat - e_buy) ** 2).sum() + \
                  (np.multiply(l_e_buy, e_buy_hat - e_buy)).sum()
        L_e_sell = 0.5 * rhos['e'] * ((e_sell_hat - e_sell) ** 2).sum() + \
                  (np.multiply(l_e_sell, e_sell_hat - e_sell)).sum()

        L_pi_buy = 0.5 * rhos['pi'] * ((pi_buy_hat - pi_buy) ** 2).sum() + \
                  (np.multiply(l_pi_buy, pi_buy_hat - pi_buy)).sum()
        L_pi_sell = 0.5 * rhos['pi'] * ((pi_sell_hat - pi_sell) ** 2).sum() + \
                  (np.multiply(l_pi_sell, pi_sell_hat - pi_sell)).sum()

        self.obj_updating_part = L_e_buy + L_e_sell + L_pi_buy + L_pi_sell

    def updateObjective(self):
        self.model.setObjective(self.obj_fixed_part + self.obj_updating_part, sense=GRB.MINIMIZE)
        self.model.update()

    def getVarValues(self):
        e_buy = np.array(self.model.getAttr('X', self.e_buy.values())).reshape((self.I, self.SH))
        e_sell = np.array(self.model.getAttr('X', self.e_sell.values())).reshape((self.I, self.SH))
        pi_buy = np.array(self.model.getAttr('X', self.pi_buy.values())).reshape((self.I, self.SH))
        pi_sell = np.array(self.model.getAttr('X', self.pi_sell.values())).reshape((self.I, self.SH))
        return e_buy, e_sell, pi_buy, pi_sell


