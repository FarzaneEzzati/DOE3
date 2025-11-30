import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np
import pickle as pkl
from itertools import product

env = gp.Env()
env.setParam("OutputFlag", 0)
env.setParam("Threads", 3)
env.setParam("TimeLimit", 120)
env.setParam("MIPGap", 0.001)
env.setParam("NumericFocus", 2)  
env.setParam("FeasibilityTol", 1e-3)  
env.setParam("Cuts", 2)

class Model:
    def __init__(self, T, N, mg_data: dict):
        T_range, N_range = range(T), range(N)
        M = 100

        self.model = gp.Model('FullModel', env=env)
        # Buy/sell mode
        self.x = self.model.addMVar((N, T), vtype=GRB.BINARY, name='x')
        # Trading variables
        self.e_b  = self.model.addMVar((N, N, T), name='e_b')  # energy traded
        self.e_s  = self.model.addMVar((N, N, T), name='e_s')  # energy traded
        self.pi_b = self.model.addMVar((N, N, T), name='pi_b')  # energy traded
        self.pi_s = self.model.addMVar((N, N, T), name='pi_s')  # energy traded
        # Device variables
        self.g_pv = self.model.addMVar((N, T), name='g_pv')   # generation by pv devices
        self.g_dg = self.model.addMVar((N, T), name='g_dg')   # generation by dg devices
        self.l_m  = self.model.addMVar((N, T), name='l_m')    # load met
        self.r_c  = self.model.addMVar((N, T), name='r_c')    # charging amount
        self.r_d  = self.model.addMVar((N, T), name='r_d')    # discharging amount
        self.e_l  = self.model.addMVar((N, T), name='e_l')    # energy level at es
        self.l_sh = self.model.addMVar((N, T), name='l_sh')   # load shed
        # Resilience and financial benefits
        self.eta_r = self.model.addMVar(N, ub=1, name='eta_r')
        self.eta_c = self.model.addMVar(N, ub=1, name='eta_c')
        # Costs
        self.C_es = self.model.addMVar(N, name='C_es')                     # Energy storage
        self.C_sh = self.model.addMVar(N, name='C_sh')                     # Shedding
        self.C_dg = self.model.addMVar(N, name='C_dg')                     # Fuel consumption  
        self.C_u  = self.model.addMVar(N, name='C_u')                      # Utility fee
        self.C_r  = self.model.addMVar(N, name='C_r')                      # Revenue
        self.C_e  = self.model.addMVar(N, lb=-float('inf'), name='C_e')    # Trade payments     
        self.C_t  = self.model.addMVar(N, lb=-float('inf'), name='C_t')    # Total cost
        for n in range(N):
            mg = mg_data[n]
            # Constraints: Hourly trades
            # Self trades
            self.model.addConstr(self.e_s[n, n].sum() + self.e_b[n, n].sum() == 0)
            for j, t in product(N_range, T_range):
                # Buy/sell mode
                self.model.addConstr(self.e_b[n, j, t] <= self.x[n, t] * M,       name=f'buy_mode[{n},{j},{t}]')
                self.model.addConstr(self.e_s[n, j, t] <= (1 - self.x[n, t]) * M, name=f'sell_mode[{n},{j},{t}]')
                # Payment Limits
                max_buy_pay = mg['pay_max'] * self.e_b[n, j, t]
                min_buy_pay = mg['pay_min'] * self.e_b[n, j, t]
                self.model.addConstr(self.pi_b[n, j, t] <= max_buy_pay, name=f'max_buy_pay[{n},{j},{t}]')
                self.model.addConstr(self.pi_b[n, j, t] >= min_buy_pay, name=f'min_buy_pay[{n},{j},{t}]')
                max_sell_pay = mg['pay_max'] * self.e_s[n, j, t]
                min_sell_pay = mg['pay_min'] * self.e_s[n, j, t]
                self.model.addConstr(self.pi_s[n, j, t] <= max_sell_pay, name=f'max_sell_pay[{n},{j},{t}]')
                self.model.addConstr(self.pi_s[n, j, t] >= min_sell_pay, name=f'min_sell_pay[{n},{j},{t}]')
                # Market constraints
                self.model.addConstr(self.e_b[n, j, t] - self.e_s[j, n, t] == 0,   name=f'e_clear[{n},{j},{t}]')
                self.model.addConstr(self.pi_b[n, j, t] - self.pi_s[j, n, t] == 0, name=f'pi_clear[{n},{j},{t}]')
            # Constraints: Scheduling
            # Start with full ES
            self.model.addConstr(self.e_l[n, 0] == mg['es_capacity'])
            for t in T_range:
                # Power generation
                self.model.addConstr(self.g_pv[n, t] <= mg['pv_capacity'] * mg['pv_hourly'][t], name=f'g_pv[{n},{t}]')
                self.model.addConstr(self.g_dg[n, t] <= mg['dg_capacity'] * mg['dg_eff'], name=f'g_dg[{n},{t}]')
                # ES function
                self.model.addConstr(self.r_c[n, t] <= mg['charge_eff'] * mg['es_capacity'], name=f'es_doc[{n},{t}]')
                self.model.addConstr(self.r_d[n, t] <= mg['discharge_eff'] * mg['es_capacity'], name=f'es_dod[{n},{t}]')
                self.model.addConstr(self.e_l[n, t] <= mg['es_capacity'], name=f'e_l[{n},{t}]')
                # Shedding amount
                load_shed = mg['load'][t] - self.l_m[n, t]
                self.model.addConstr(self.l_sh[n, t] == load_shed, name=f'load_shed[{n},{t}]')
                # Power input/output
                input = self.g_pv[n, t] + self.g_dg[n, t] + self.r_d[n, t] + self.e_b[n, :, t].sum()
                output = self.l_m[n, t] + self.r_c[n, t] + self.e_s[n, :, t].sum()
                self.model.addConstr(output == input, name=f'balance[{n},{t}]')
                # ES level flow
                if t < T-1:
                    next_es_level = self.e_l[n, t] +\
                                    self.r_c[n, t] * mg['charge_eff'] -\
                                    self.r_d[n, t] / mg['discharge_eff']
                    self.model.addConstr(self.e_l[n, t + 1] == next_es_level, name=f'flow[{n},{t}]')
            # Costs: ES usage
            C_es = mg['es_cost'] * (self.r_c[n] + self.r_d[n]).sum()
            self.model.addConstr(self.C_es[n] == C_es)
            # Costs: DG usage
            C_dg = mg['dg_cost'] * self.g_dg[n].sum()
            self.model.addConstr(self.C_dg[n] == C_dg)
            # Costs: Shedding
            C_sh = mg['shed_cost'] * self.l_sh[n].sum()
            self.model.addConstr(self.C_sh[n] == C_sh)
            # Costs: utility fees
            C_u = (1 - mg['usrr']) * mg['u_cost'] * (self.e_b[n]+ self.e_s[n]).sum()
            self.model.addConstr(self.C_u[n] == C_u)
            # Costs: trade payments
            C_e = (1 - mg['fsrr']) * self.pi_b[n].sum() - self.pi_s[n].sum()
            self.model.addConstr(self.C_e[n] == C_e)
            # Costs: revenue of seeling to households
            C_r = mg['load_price'] * self.l_m[n].sum()
            self.model.addConstr(self.C_r[n] == C_r)
            # Costs: total cost
            C_t = self.C_es[n] + self.C_dg[n] + self.C_sh[n] + self.C_u[n] + self.C_e[n] - self.C_r[n]
            self.model.addConstr(self.C_t[n] == C_t)
        
            # Energy equity
            eta_r = self.l_m[n].sum() / mg['load'][:T].sum() 
            eta_c = 1 - (self.C_t[n] - mg['C_t_min']) / (mg['C_t_max'] - mg['C_t_min'])
            self.model.addConstr(self.eta_r[n] == eta_r, name=f'eta_r{n}')
            self.model.addConstr(self.eta_c[n] == eta_c, name=f'eta_c{n}')
        
        # Objective
        self.objective = -sum((mg_data[n]['alpha'] * (mg_data[n]['r_priority'] * self.eta_r[n] + mg_data[n]['c_priority'] * self.eta_c[n]) 
                              for n in N_range))
        self.model.setObjective(self.objective, sense=GRB.MINIMIZE)
        
        # Subsidies
        fs_used = sum(mg_data[n]['fsrr'] * self.pi_b[n].sum() for n in N_range)
        us_used = sum(mg_data[n]['u_cost'] * mg_data[n]['usrr'] * (self.e_b[n] + self.e_s[n]).sum() for n in N_range)
        self.model.addConstr(fs_used <= mg_data[0]['TFS'], name='TFS')
        self.model.addConstr(us_used <= mg_data[0]['TUS'], name='TUS')
        
        # Trade off baseline
        trade_off_constr = self.model.addConstr(self.e_b.sum() + self.e_s.sum() == 0, name='trade_off')
        self.model.optimize()
        self.eta_r_Non = self.eta_r.x
        self.eta_c_Non = self.eta_c.x
        self.C_t_Non = self.C_t.x
        self.model.remove(trade_off_constr)

        for n in N_range:
            self.model.addConstr(self.eta_r[n] >= mg_data[n]['tau'] * self.eta_r_Non[n])
            self.model.addConstr(self.eta_c[n] >= mg_data[n]['tau'] * self.eta_c_Non[n])
        self.model.update()
    
    def get_var_values(self):
        keys = ['x',
            'e_b', 'e_s', 'pi_b', 'pi_s', 
            'eta_r', 'eta_c',
            'C_sh', 'C_es', 'C_r', 'C_u', 'C_t', 'C_e', 'C_dg']
        result = {k: getattr(self, k).x for k in keys}
        result['obj'] = -self.model.ObjVal
        result['eta_r_Non'] = self.eta_r_Non
        result['eta_c_Non'] = self.eta_c_Non
        return result



