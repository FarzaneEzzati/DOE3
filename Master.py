import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

class Master:
    def __init__(self, data):
        self.N = data['N']
        self.T = data['T']
        self.fsrr = data['fsrr']
        self.usrr = data['usrr']
        self.TFS = data['TFS']
        self.TUS = data['TUS']
        self.u_cost = data['u_cost']

        self.model = gp.Model('Master')
        self.model.setParam('Method', 0)
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('NumericFocus', 1)
        self.model.setParam('FeasibilityTol', 1e-3)
        self.model.setParam('Cuts', 3)
        self.model.setParam('MIPGap', 1e-4)

        # auxiliary variables e_hat, pi_hat
        self.e_buy_hat = self.model.addMVar((self.N, self.N, self.T), name='e_buy_hat')
        self.e_sell_hat = self.model.addMVar((self.N, self.N, self.T), name='e_sell_hat')
        self.pi_buy_hat = self.model.addMVar((self.N, self.N, self.T), name='pi_buy_hat')
        self.pi_sell_hat = self.model.addMVar((self.N, self.N, self.T), name='pi_sell_hat')
        self.fsrr_slack = self.model.addVar(name='fsrr_slack')
        self.usrr_slack = self.model.addVar(name='usrr_slack')

        stacked_trades = np.stack((self.e_buy_hat, self.e_sell_hat, self.pi_buy_hat, self.pi_sell_hat), axis=0)
        self.yhat = np.array([mvar.tolist() for mvar in stacked_trades], dtype=object)

        # constraints
        for i in range(self.N):
            self.model.addConstr(self.e_sell_hat[i, i].sum() + self.e_buy_hat[i, i].sum() == 0, name='self_trade')
            for j in range(self.N):
                for t in range(self.T):
                    self.model.addConstr(self.e_buy_hat[i, j, t] - self.e_sell_hat[j, i, t] == 0, name='e_clear')
                    self.model.addConstr(self.pi_buy_hat[i, j, t] - self.pi_sell_hat[j, i, t] == 0, name='pi_clear')

        self.model.addConstr(
            sum(self.pi_buy_hat[i] * self.fsrr[i] for i in range(self.N)) +
            self.fsrr_slack == self.TFS, name='TFS')
        self.model.addConstr(
            sum(self.u_cost * self.usrr[i] * (self.e_buy_hat[i] + self.e_sell_hat[i]) for i in range(self.N)) +
            self.usrr_slack == self.TUS, name='TUS')
        self.model.update()

        self.obj_lagrangian = 0


    def solve_with(self, y, rho_y, l_y):
        y_minus_yhat = y - self.yhat
        lag_y = l_y * y_minus_yhat + 0.5 * rho_y * y_minus_yhat ** 2
        lag_y_sum = lag_y.sum()

        self.obj_lagrangian = lag_y_sum
        self.model.setObjective(self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        # solve model
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return z is model optimal or timed out, o.w. interrupt
        if self.model.status == 2:
            return self.get_yhat_opt()
        else:
            print(self.model.Status)
            raise ValueError('Master stopped.')

    def get_yhat_opt(self):
        return np.stack((self.e_buy_hat.x,
                         self.e_sell_hat.x,
                         self.pi_buy_hat.x,
                         self.pi_sell_hat.x))
