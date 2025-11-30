import pickle
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
from MGConfig import MG_config
from itertools import product

env = gp.Env()
env.setParam("OutputFlag", 0)
env.setParam("Threads", 3)
env.setParam("TimeLimit", 60)
env.setParam("MIPGap", 0.01)
env.setParam("NumericFocus", 3)  
env.setParam("FeasibilityTol", 1e-4)  
env.setParam("Cuts", 2)

class Sub:
    def __init__(self, T, N, data, MIP=False):
        n = data['n']
        T_range, N_range = range(T), range(N)
        M = 100

        self.model = gp.Model(f'MG({n})', env=env)
        # x variables
        if MIP:
            self.x = self.model.addVars(T, vtype=GRB.BINARY, name='u')
        else:
            self.x = self.model.addMVar(T, lb=0, ub=1, name='x')
            env.setParam("Method", 2)
        # Trading variables
        self.e_b = self.model.addMVar((N, T), name='e_b')
        self.e_s = self.model.addMVar((N, T), name='e_s')
        self.pi_b = self.model.addMVar((N, T), name='pi_b')
        self.pi_s = self.model.addMVar((N, T), name='pi_s')
        # Device variables
        self.g_pv = self.model.addMVar(T, name='g_pv')
        self.g_dg =  self.model.addMVar(T, name='g_dg')
        self.l_m =  self.model.addMVar(T, name='l_m')
        self.r_c =  self.model.addMVar(T, name='r_c')
        self.r_d =  self.model.addMVar(T, name='r_d')
        self.e_l =  self.model.addMVar(T, name='e_l')
        self.l_sh =  self.model.addMVar(T, name='l_sh')
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
        
        # Constraints: Hourly trades
        self.model.addConstr((self.e_s[n] + self.e_b[n]).sum() == 0, name=f'self_trade')
        # Hourly trade
        for j, t in product(N_range, T_range):
            # Buy/sell mode
            self.model.addConstr(self.e_b[j, t] <= self.x[t] * M,       name=f'buy_mode[{j},{t}]')
            self.model.addConstr(self.e_s[j, t] <= (1 - self.x[t]) * M, name=f'sell_mode[{j},{t}]')
            # Payment Limits
            max_buy_pay = data['pay_max'] * self.e_b[j, t]
            min_buy_pay = data['pay_min'] * self.e_b[j, t]
            self.model.addConstr(self.pi_b[j, t] <= max_buy_pay,  name=f'buy_pay_max[{j},{t}]')
            self.model.addConstr(self.pi_b[j, t] >= min_buy_pay,  name=f'buy_pay_min[{j},{t}]')
            max_sell_pay = data['pay_max'] * self.e_s[j, t]
            min_sell_pay = data['pay_min'] * self.e_s[j, t]
            self.model.addConstr(self.pi_s[j, t] <= max_sell_pay, name=f'sell_pay_max[{j},{t}]')
            self.model.addConstr(self.pi_s[j, t] >= min_sell_pay, name=f'sell_pay_min[{j},{t}]')
 
        # Constraints: Scheduling
        # Start with full ES
        self.model.addConstr(self.e_l[0] == data['es_capacity'])
        for t in T_range:
            # Power generation
            self.model.addConstr(self.g_pv[t] <= data['pv_capacity'] * data['pv_hourly'][t], name=f'g_pv[{t}]')
            self.model.addConstr(self.g_dg[t] <= data['dg_capacity'] * data['dg_eff'],       name=f'g_dg[{t}]')
            # ES function
            self.model.addConstr(self.r_c[t] <= data['charge_eff'] * data['es_capacity'],     name=f'es_doc[{t}]')
            self.model.addConstr(self.r_d[t] <= data['discharge_eff'] * data['es_capacity'],  name=f'es_dod[{t}]')
            self.model.addConstr(self.e_l[t] <= data['es_capacity'], name=f'e_l[{t}]')
            # Shedding amount
            self.model.addConstr(self.l_sh[t] == data['load'][t] - self.l_m[t], name=f'load_shed[{t}]')
            # Power input/output
            input = self.g_pv[t] + self.g_dg[t] + self.r_d[t] + self.e_b[:, t].sum()
            output = self.l_m[t] + self.r_c[t] + self.e_s[:, t].sum()
            self.model.addConstr(input == output, name=f'balance[{t}]')
            if t < T-1:
                es_level_next = self.e_l[t] + self.r_c[t] * data['charge_eff'] - self.r_d[t] / data['discharge_eff']
                self.model.addConstr(self.e_l[t + 1] == es_level_next, name=f'flow[{t}]')
    
        # Costs: ES usage
        C_es = data['es_cost'] * (self.r_c + self.r_d).sum() 
        self.model.addConstr(self.C_es == C_es, name='C_es')
        # Costs: DG usage
        C_dg = data['dg_cost'] * self.g_dg.sum()
        self.model.addConstr(self.C_dg == C_dg, name='C_dg')
        # Costs: Shedding
        C_sh = data['shed_cost'] * self.l_sh.sum()
        self.model.addConstr(self.C_sh == C_sh, name='C_ls')
        # Costs: utility fees
        C_u = (1 - data['usrr']) * data['u_cost'] * (self.e_b + self.e_s).sum()
        self.model.addConstr( self.C_u == C_u, name='C_u')
        # Costs: trade payments
        C_e = (1 - data['fsrr']) * self.pi_b.sum() - self.pi_s.sum()
        self.model.addConstr(self.C_e == C_e, name='C_e')
        # Costs: revenue of seeling to households
        C_r = data['load_price'] * self.l_m.sum()
        self.model.addConstr(self.C_r == C_r, name='C_r')
        # Costs: total cost
        C_t = self.C_es + self.C_dg + self.C_sh + self.C_u + self.C_e - self.C_r
        self.model.addConstr(self.C_t == C_t, name='C_t')
        
        # Energy equity
        eta_r = self.l_m.sum() / data['load'][:T].sum()
        eta_c = 1 - (self.C_t - data['C_t_min']) / (data['C_t_max'] - data['C_t_min'])
        self.model.addConstr(self.eta_r == eta_r, name='eta_r')
        self.model.addConstr(self.eta_c == eta_c, name='eta_c')

        # Objective 
        self.obj_fixed = -(data['alpha'] * (data['r_priority'] * self.eta_r + data['c_priority'] * self.eta_c))
        self.model.setObjective(self.obj_fixed, sense=GRB.MINIMIZE)

        # Set lower bound for resilience and cost
        trade_off_constr = self.model.addConstr((self.e_b + self.e_s).sum() == 0, name='trade_off')
        self.model.optimize()

        if self.model.SolCount < 0:
            self.show_infeasible_const()
        self.eta_r_Non = self.eta_r.x
        self.eta_c_Non = self.eta_c.x
        self.C_t_Non = self.C_t.x
        self.model.remove(trade_off_constr)

        self.model.addConstr(self.eta_r >= data['tau'] * self.eta_r_Non, name='r_improve')
        self.model.addConstr(self.eta_c >= data['tau'] * self.eta_c_Non, name='c_improve')
        self.model.update()

        self.obj_lagrangian = 0
        stacked_trades = np.stack((self.e_b, self.e_s, self.pi_b, self.pi_s), axis=0)
        self.y = np.array([mvar.tolist() for mvar in stacked_trades], dtype=object)
        self.n = n

    def solve_with_var_x(self, yhat, l_y, p_y, z, l_z, p_z):
        # yhat part
        lag_y = l_y * (self.y) + 0.5 * p_y * (self.y ** 2 - 2 * self.y * yhat ) 
        lag_y = lag_y.sum()
        
        # x part
        lag_z = l_z * (self.x) + 0.5 * p_z * (self.x ** 2 - 2 * self.x * z )
        lag_z = lag_z.sum()

        # Objective
        self.obj_lagrangian = lag_y + lag_z
        self.model.setObjective(1e6 * self.obj_fixed + self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return y is model optimal or timed out, o.w. interrupt
        if self.model.SolCount > 0:
            y_opt = np.stack([self.e_b.x, self.e_s.x, self.pi_b.x, self.pi_s.x])
            x_opt = self.x.x
            return y_opt, x_opt
        else:
            print(f'{self.model.ModelName} failed with status {self.model.Status}.')

    def solve_with_fixed_x(self, yhat, l_y, p_y):
        # yhat part
        lag_y = l_y * (self.y) + 0.5 * p_y * (self.y ** 2 - 2 * self.y * yhat ) 
        self.obj_lagrangian = lag_y.sum()
        # Objective
        self.model.setObjective(1e6 * self.obj_fixed + self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return y is model optimal or timed out, o.w. interrupt
        if self.model.SolCount > 0:
            y_opt = np.stack([self.e_b.x, self.e_s.x, self.pi_b.x, self.pi_s.x])
            return y_opt
        else:
            print(f'{self.model.ModelName} failed with status {self.model.Status}.')
    
    def show_infeasible_const(self):
        self.model.computeIIS()
        for c in self.model.getConstrs():
            if c.IISConstr:
                print(f'{self.config.id} infeasible constr: ', c.ConstrName)





