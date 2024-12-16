import gurobipy as gp
from gurobipy import GRB, quicksum
from itertools import product
import numpy as np
import copy

class HigherLevelModel:
    def __init__(self, **kwargs):
        I = kwargs['mg_count']
        SH = kwargs['scheduling_horizon']
        i_list = range(I)
        t_list = range(SH)
        self.done = False
        self.e_buy_gaps = []
        self.e_sell_gaps = []
        self.pi_buy_gaps = []
        self.pi_sell_gaps = []

        self.model = gp.Model('HigherLevel')
        self.model.setParam('OutputFlag', 0)
        # auxiliary variables e_hat, pi_hat
        self.e_buy_hat = self.model.addVars(I, I, SH, lb=0, name='e_buy_hat')
        self.e_sell_hat = self.model.addVars(I, I, SH, lb=0, name='e_buy_hat')
        self.pi_buy_hat = self.model.addVars(I, I, SH, lb=0, name='e_buy_hat')
        self.pi_sell_hat = self.model.addVars(I, I, SH, lb=0, name='e_buy_hat')

        # constraints
        self.model.addConstrs(
            (self.e_buy_hat[(i, j, t)] == self.e_sell_hat[(j, i, t)] for i in i_list for j in i_list for t in t_list),
            name='trade clear')
        self.model.addConstrs(
            (self.pi_buy_hat[(i, j, t)] == self.pi_sell_hat[(j, i, t)] for i in i_list for j in i_list for t in t_list),
            name='payment clear')
        self.model.update()
        self.I = I
        self.SH = SH
    def buildObjective(self, e_values, pi_values, lambdas, rhos):
        e_buy, e_sell = e_values['e_buy'], e_values['e_sell']
        pi_buy, pi_sell = pi_values['pi_buy'], pi_values['pi_sell']

        l_e_buy, l_e_sell = lambdas['l_e_buy'], lambdas['l_e_sell']
        l_pi_buy, l_pi_sell = lambdas['l_pi_buy'], lambdas['l_pi_sell']

        e_buy_hat = np.array(self.e_buy_hat.values()).reshape(self.I, self.I, self.SH)
        e_sell_hat = np.array(self.e_sell_hat.values()).reshape(self.I, self.I, self.SH)
        pi_buy_hat = np.array(self.pi_buy_hat.values()).reshape(self.I, self.I, self.SH)
        pi_sell_hat = np.array(self.pi_sell_hat.values()).reshape(self.I, self.I, self.SH)

        L_e_buy_part = 0.5 * rhos['e'] * ((e_buy_hat - e_buy) ** 2).sum() + \
                 np.multiply(l_e_buy, e_buy_hat - e_buy).sum()
        L_e_sell_part = 0.5 * rhos['e'] * np.sum((e_sell_hat - e_sell) ** 2) + \
                  np.multiply(l_e_sell, e_sell_hat - e_sell).sum()

        L_pi_buy_part = 0.5 * rhos['pi'] * ((pi_buy_hat - pi_buy) ** 2).sum() + \
                  np.multiply(l_pi_buy, pi_buy_hat - pi_buy).sum()
        L_pi_sell_part = 0.5 * rhos['pi'] * ((pi_sell_hat - pi_sell) ** 2).sum() + \
                  np.multiply(l_pi_sell, pi_sell_hat - pi_sell).sum()
        self.model.setObjective(L_e_buy_part + L_e_sell_part +
                                L_pi_buy_part + L_pi_sell_part, sense=GRB.MINIMIZE)
        self.model.update()
    def updateLambdas(self, e_values, pi_values, e_hat_values, pi_hat_values, lambdas, rhos):
        e_buy, e_sell = e_values['e_buy'], e_values['e_sell']
        pi_buy, pi_sell = pi_values['pi_buy'], pi_values['pi_sell']

        e_buy_hat, e_sell_hat = e_hat_values['e_buy'], e_hat_values['e_sell']
        pi_buy_hat, pi_sell_hat = pi_hat_values['pi_buy'], pi_hat_values['pi_sell']

        lambdas['l_e_buy'] +=  rhos['e'] * (e_buy_hat - e_buy)
        lambdas['l_e_sell'] += rhos['e'] * (e_sell_hat - e_sell)
        lambdas['l_pi_buy'] += rhos['pi'] * (pi_buy_hat - pi_buy)
        lambdas['l_pi_sell'] += rhos['pi'] * (pi_sell_hat - pi_sell)
        return lambdas

    def getVarValues(self):
        e_hat_values = {'e_buy': None, 'e_sell': None}
        pi_hat_values = {'pi_buy': None, 'pi_sell': None}
        e_hat_values['e_buy'] = np.array(self.model.getAttr('X', self.e_buy_hat.values())).reshape(
            (self.I, self.I, self.SH))
        e_hat_values['e_sell'] = np.array(self.model.getAttr('X', self.e_sell_hat.values())).reshape(
            (self.I, self.I, self.SH))
        pi_hat_values['pi_buy'] = np.array(self.model.getAttr('X', self.pi_buy_hat.values())).reshape(
            (self.I, self.I, self.SH))
        pi_hat_values['pi_sell'] = np.array(self.model.getAttr('X', self.pi_sell_hat.values())).reshape(
            (self.I, self.I, self.SH))
        return e_hat_values, pi_hat_values


