import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from itertools import product

env = gp.Env()
env.setParam('OutputFlag', 0)
env.setParam('Method', 2)
env.setParam("NumericFocus", 3) 

class Master:
    def __init__(self, T, N, data):
        N_range, T_range = range(N), range(T)

        self.model = gp.Model(env=env)
        # Variables e_hat, pi_hat
        self.e_b_hat = self.model.addMVar((N, N, T), name='e_b_hat')
        self.e_s_hat = self.model.addMVar((N, N, T), name='e_s_hat')
        self.pi_b_hat = self.model.addMVar((N, N, T), name='pi_b_hat')
        self.pi_s_hat = self.model.addMVar((N, N, T), name='pi_s_hat')
        stacked_trades = np.stack((self.e_b_hat, self.e_s_hat, self.pi_b_hat, self.pi_s_hat), axis=0)
        self.yhat = np.array([mvar.tolist() for mvar in stacked_trades], dtype=object)
        # Market 
        for n in N_range:
            self.model.addConstr(self.e_s_hat[n, n].sum() + self.e_b_hat[n, n].sum() == 0, name='self_trade')
            for j, t in product(N_range, T_range):
                self.model.addConstr(self.e_b_hat[n, j, t] - self.e_s_hat[j, n, t] == 0, name='e_clear')
                self.model.addConstr(self.pi_b_hat[n, j, t] - self.pi_s_hat[j, n, t] == 0, name='pi_clear')
        # Subsidy
        fs_used = sum(data['fsrr'][n] * self.pi_b_hat[n].sum()  for n in N_range)
        us_used = sum(data['u_cost'] * data['fsrr'][n] * (self.e_b_hat[n] + self.e_s_hat[n]).sum() for n in N_range)
        self.model.addConstr(fs_used <= data['TFS'], name='TFS')
        self.model.addConstr(us_used <= data['TUS'], name='TUS')
        self.model.update()

        self.obj_lagrangian = 0


    def solve_with(self, y, p_y, l_y):
        lag_y = - l_y * self.yhat + 0.5 * p_y * (self.yhat ** 2 - 2 * self.yhat * y)
        lag_y = lag_y.sum()

        self.obj_lagrangian = lag_y
        self.model.setObjective(self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        # solve model
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return z is model optimal or timed out, o.w. interrupt
        if self.model.SolCount > 0:
            y_opt = np.stack([self.e_b_hat.x,self.e_s_hat.x,self.pi_b_hat.x,self.pi_s_hat.x])
            return y_opt
        else:
            raise ValueError(f'Master stopped with status {self.model.Status}')
