"""
The subproblems for ADMM are defined here.
"""

import pickle
import gurobipy as gp
from gurobipy import GRB, quicksum
import numpy as np

status = {1: 'LOADED',
         2: 'OPTIMAL',
         3: 'INFEASIBLE',
         4: 'INF_OR_UNBD',
         5: 'UNBOUNDED',
         6: 'CUTOFF',
         7: 'ITERATION_LIMIT',
         8: 'NODE_LIMIT',
         9: 'TIME_LIMIT',
         10: 'SOLUTION_LIMIT',
         11: 'INTERRUPTED',
         12: 'NUMERIC',
         13: 'SUBOPTIMAL',
         14: 'INPROGRESS',
         15: 'USER_OBJ_LIMIT'}


class Sub:
    def __init__(self, data, trade, MIP=False, PWL=False):
        # Initialize
        self.fsrr = data['fsrr']  # financial support rate
        self.usrr = data['usrr']  # utility support rate
        self.C_t_min = data['C_t_min']
        self.C_t_max = data['C_t_max']
        self.id = data['id']
        self.I = data['n_mgs']
        self.HL = data['HL']
        self.S = data['n_scen']
        self.alpha = data['alpha']
        self.theta_r = data['theta_r']
        self.theta_c = data['theta_c']
        self.mg_id = data['id']
        self.sv = data['sv']
        self.trade = trade
        self.tau_r = 0.9
        self.tau_c = 0.99
        self.eta_r_Non = None
        self.eta_c_Non = None
        self.M = 1e3

        self.model = gp.Model()
        self.model.setParam("OutputFlag", 0)
        self.model.setParam("Threads", 3)
        self.model.setParam("TimeLimit", 60)
        self.model.setParam("MIPGap", 0.1)
        self.model.setParam("NumericFocus", 3)  # Prioritize numerical stability
        self.model.setParam("FeasibilityTol", 1e-2)  # Increase feasibility tolerance
        self.model.setParam("Cuts", 2)
        self.model.setParam("NonConvex", 2)

        # Device variables
        self.g_pv = self.addVar('g_pv', self.S, self.HL)  # generation by pv devices
        self.g_dg = self.addVar('g_dg', self.S, self.HL)  # generation by dg devices
        self.d = self.addVar('x', self.S, self.HL)      # load served
        self.r_c = self.addVar('r_c', self.S, self.HL)  # charging amount
        self.r_d = self.addVar('r_d', self.S, self.HL)  # discharging amount
        self.el = self.addVar('el', self.S, self.HL)      # energy level at es
        self.ls = self.addVar('ls', self.S, self.HL)    # load shed
        # Trading variables
        self.e_buy = self.addVar('e_buy', self.I, self.HL)  # energy traded
        self.e_sell = self.addVar('e_sell', self.I, self.HL)    # energy traded
        self.pi_buy = self.addVar('pi_buy', self.I, self.HL)  # energy traded
        self.pi_sell = self.addVar('pi_sell', self.I, self.HL)    # energy traded
        # Buy/sell mode
        if MIP:
            self.u = np.array([self.model.addVar(vtype=GRB.BINARY, name='u') for _ in range(self.HL)])
        else:
            self.u = np.array([self.model.addVar(lb=-10, ub=10, name='u') for _ in range(self.HL)])
        # Resilience and financial benefits
        self.eta_r = self.model.addVar(ub=1, name='eta_r')
        self.eta_c = self.model.addVar(ub=1, name='eta_c')
        # Costs
        self.C_es = self.model.addVar(lb=0, name='C_es')  # es operation cost
        self.C_ls = self.model.addVar(lb=0, name='C_ls')  # load shedding cost
        self.C_dg = self.model.addVar(lb=0, name='C_dg')  # dg operation cost
        self.C_u = self.model.addVar(lb=0, name='C_u')    # utility cost
        self.C_e = self.model.addVar(lb=-float('inf'), name='C_e')  # trade payments
        self.C_t = self.model.addVar(lb=-float('inf'), name='C_t')  # total cost
        # Vars for admm
        self.y = np.stack((self.e_buy, self.e_sell, self.pi_buy, self.pi_sell), axis=0)
        # Constraints
        self.addConstraints(data)
        # Objective function
        self.obj_original = -(self.alpha * (self.theta_r * self.eta_r + self.theta_c * self.eta_c))
        self.model.setObjective(self.obj_original, sense=GRB.MINIMIZE)
        # Set lower bound for resilience and cost
        if self.trade:
            trade_off_constr = self.model.addConstr(np.sum(self.e_buy + self.e_sell) == 0, name='trade_off')
            self.model.optimize()

            if self.model.status == GRB.INFEASIBLE:
                self.show_infeasible_const()

            self.eta_r_Non = self.eta_r.x
            self.eta_c_Non = self.eta_c.x

            self.model.remove(trade_off_constr)
            tshd_r = self.tau_r * (1 - self.sv) + self.sv
            tshd_c = self.tau_c * (1 - self.sv) + self.sv
            self.model.addConstr(-self.eta_r <= - tshd_r * self.eta_r_Non, name='res_improve')
            self.model.addConstr(-self.eta_c <= - tshd_c * self.eta_c_Non, name='cost_improve')
            self.model.update()
        else:
            self.model.addConstr(np.sum(self.e_buy) + np.sum(self.e_sell) == 0, name='trade_off')
        self.model.update()
        self.obj_lagrangian = 0


    def solve_with(self, yhat, z1, z2, l_yhat, l_z1, l_z2, rho_yhat, rho_z1, rho_z2, xi):
        lag_y = 0.5 * rho_yhat * (self.y - (rho_yhat * yhat - l_yhat) / rho_yhat) ** 2
        lag_y_sum = lag_y.sum()

        lag_x = 0.5 * (rho_z1 + rho_z2) * (self.u - (rho_z1 * z1 + rho_z2 * z2 - l_z1 - l_z2) / (rho_z1 + rho_z2)) ** 2
        lag_x_sum = lag_x.sum()

        self.obj_lagrangian = xi * (lag_x_sum + lag_y_sum)
        self.model.setObjective(self.obj_original + int(self.trade) * self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()

        # solve model
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return y is model optimal or timed out, o.w. interrupt
        if self.model.Status == GRB.OPTIMAL or self.model.SolCount > 0:
            y_opt = self.get_y_opt()
            u_opt = self.get_u_opt()
            return y_opt, u_opt
        elif self.model.Status == GRB.INFEASIBLE:
            print(f'sub-problem {self.mg_id} infeasible. Infeasible constraints: ')
            self.show_infeasible_const()
            raise ValueError('Algorithm stopped.')


    def addVar(self, name, dim1=0, dim2=0, dim3=0, dim4=0, dim5=0, ub=float('inf')):
        assert dim1 > 0, "First dimension must be > 0."

        # Determine the shape dynamically
        shape = tuple(dim for dim in [dim1, dim2, dim3, dim4, dim5] if dim > 0)

        # Create an empty NumPy array to store variables
        var_matrix = np.empty(shape, dtype=object)

        # Iterate over all possible index combinations and create variables
        for index in np.ndindex(*shape):
            var_name = f"{name}{tuple(index)}"  # Ensures readable names
            var_matrix[index] = self.model.addVar(name=var_name, ub=ub)

        return var_matrix


    def get_y_opt(self):
        return np.array([var.X for var in self.y.flatten()]).reshape(self.y.shape)

    def get_u_opt(self):
        return np.array([var.X for var in self.u])

    @staticmethod
    def get_var_X(variable):
        return np.array([var.X for var in variable.flatten()]).reshape(variable.shape)

    def show_infeasible_const(self):
        self.model.computeIIS()
        for c in self.model.getConstrs():
            if c.IISConstr:
                print('infeasible constr: ', c.ConstrName)

    def save_solutions(self, final_l=None, final_z=None, final_rho=None, xi=None, identifier=None):
        if self.trade:
            _ = self.solve_with(final_l, final_z, final_rho, xi)
        else:
            self.model.optimize()

        g_pv = self.get_var_X(self.g_pv)
        g_dg = self.get_var_X(self.g_dg)
        x = self.get_var_X(self.d)
        r_c = self.get_var_X(self.r_c)
        r_d = self.get_var_X(self.r_d)
        el = self.get_var_X(self.el)
        ls = self.get_var_X(self.ls)
        # trading variables
        e_sell = self.get_var_X(self.e_sell)
        e_buy = self.get_var_X(self.e_buy)
        pi_sell = self.get_var_X(self.pi_sell)
        pi_buy = self.get_var_X(self.pi_buy)
        u = self.get_var_X(self.u)
        # equity metrics
        eta_r, eta_c = self.eta_r.x, self.eta_c.x
        eta_r_Non, eta_c_Non = self.eta_r_Non, self.eta_c_Non
        # costs
        C_es, C_ls, C_dg, C_u = self.C_es.x, self.C_ls.x, self.C_dg.x, self.C_u.x
        C_e, C_t = self.C_e.x, self.C_t.x

        with open(f'Solutions/{identifier}MG({self.mg_id}).pkl', 'wb') as handle:
            pickle.dump([final_l, final_z, final_rho,
                         g_pv, g_dg, x, r_c, r_d, el, ls,
                         e_sell, e_buy, pi_sell, pi_buy, u,
                         eta_r, eta_c, eta_r_Non, eta_c_Non,
                         C_es, C_ls, C_dg, C_u, C_e, C_t], handle)
        handle.close()

    def addConstraints(self, data):
        self.model.addConstrs(
            (self.g_pv[s, t] <= data['pv'] * data['pv_s'][s, t] for s, t in np.ndindex(self.S, self.HL)),
            name=f'g_pv_limit')
        self.model.addConstrs(
            (self.g_dg[s, t] <= data['dg'] * data['dg_ef'] for s, t in np.ndindex(self.S, self.HL)),
            name=f'g_dg_limit')
        self.model.addConstrs(
            (self.d[s, t] <= data['l_s'][s, t] for s, t in np.ndindex(self.S, self.HL)),
            name=f'x_limit')
        self.model.addConstrs(
            (self.r_c[s, t] <= data['doc'] * data['es'] for s, t in np.ndindex(self.S, self.HL)),
            name=f'es_doc')
        self.model.addConstrs(
            (self.r_d[s, t] <= data['dod'] * data['es'] for s, t in np.ndindex(self.S, self.HL)),
            name=f'es_dod')
        self.model.addConstrs(
            (self.el[s, t] <= data['es'] for s, t in np.ndindex(self.S, self.HL)),
            name=f'l limit')
        self.model.addConstrs(
            (self.ls[s, t] == data['l_s'][s, t] - self.d[s, t] for s, t in np.ndindex(self.S, self.HL)),
            name='load shed')
        self.model.addConstrs(
            (self.d[s, t] + self.r_c[s, t] + sum(self.e_sell[j, t] for j in range(self.I)) ==
             self.g_pv[s, t] + self.g_dg[s, t] + self.r_d[s, t] + sum(self.e_buy[:, t])
             for s, t in np.ndindex(self.S, self.HL)),
            name=f'balance')
        self.model.addConstrs(
            (self.el[s, 0] == data['es'] for s in range(self.S)), name=f'l0')
        self.model.addConstrs(
            (self.el[s, t + 1] == self.el[s, t] + self.r_c[s, t] * data['es_c'] -
             self.r_d[s, t] / data['es_d'] for s, t in np.ndindex(self.S, self.HL - 1)),
            name=f'flow')
        # Trade constraints
        self.model.addConstrs(
            (self.e_sell[self.id, t] + self.e_buy[self.id, t] == 0 for t in range(self.HL)),
            name=f'self')
        self.model.addConstrs(
            (self.e_buy[j, t] <= self.u[t] * self.M for j, t in np.ndindex(self.I, self.HL)),
            name='buy mode')
        self.model.addConstrs(
            (self.e_sell[j, t] <= (1 - self.u[t]) * self.M for j, t in np.ndindex(self.I, self.HL)),
            name='sell mode')
        self.model.addConstrs(
            (self.pi_buy[j, t] <= data['pay_max'] * self.e_buy[j, t] for j, t in np.ndindex(self.I, self.HL)),
            name='buy pay max')
        self.model.addConstrs(
            (-self.pi_buy[j, t] <= -data['pay_min'] * self.e_buy[j, t] for j, t in np.ndindex(self.I, self.HL)),
            name='buy pay min')
        self.model.addConstrs(
            (self.pi_sell[j, t] <= data['pay_max'] * self.e_sell[j, t] for j, t in np.ndindex(self.I, self.HL)),
            name='sell pay max')
        self.model.addConstrs(
            (-self.pi_sell[j, t] <= -data['pay_min'] * self.e_sell[j, t] for j, t in np.ndindex(self.I, self.HL)),
            name='sell pay min')
        # Costs
        self.model.addConstr(
            self.C_es == data['es_cost'] *
            np.sum(np.average(self.r_c + self.r_d, weights=data['probs'], axis=0)), name='C_es')
        self.model.addConstr(
            self.C_dg == data['dg_cost'] *
            np.sum(np.average(self.g_dg, weights=data['probs'], axis=0)), name='C_dg')
        self.model.addConstr(
            self.C_ls == data['lsp'] *
            np.sum(np.average(self.ls, weights=data['probs'], axis=0)), name='C_dg')
        self.model.addConstr(
            self.C_u == (1 - self.usrr) * data['u_cost'] * np.sum(self.e_buy + self.e_sell), name='C_u')
        self.model.addConstr(
            self.C_e == (1 - self.fsrr) * np.sum(self.pi_buy) - np.sum(self.pi_sell), name='C_e')
        self.model.addConstr(
            self.C_t == self.C_es + self.C_dg + self.C_ls + self.C_u + self.C_e, name='C_t')
        # Energy equity
        self.model.addConstr(
            self.eta_r == np.sum(1 / self.HL * np.average(self.d / data['l_s'], weights=data['probs'], axis=0)),
            name='eta_r')
        self.model.addConstr(
            self.eta_c == 1 - (self.C_t - self.C_t_min) / (self.C_t_max - self.C_t_min), name='eta_c')
        self.model.update()

    """def evaluate_Q_ill(self, rho, xi):
        Q_mat = 0.5 * rho * xi * np.diag((1 / (self.y_max ** 2)).flatten())
        Q_cond_number = np.linalg.cond(Q_mat)
        Q_min_eigen = np.min(np.linalg.eigvals(Q_mat))
        ill_condition = Q_cond_number >= 1e6 and Q_min_eigen <= 1e-6
        return ill_condition  # returns true if Q is ill conditioned

    def reform_as_PWL(self, n_segments, y_min, y_max):
        dim1, dim2, dim3, dim4 = 4, self.I, self.HL, n_segments

        # uniform breakpoints and f_values
        y_values = np.zeros((dim1, dim2, dim3, dim4))
        for d1, d2, d3 in np.ndindex(dim1, dim2, dim3):
            y_values[d1, d2, d3, :] = np.linspace(y_min[d1, d2, d3], y_max[d1, d2, d3], dim4)
        f_values = y_values ** 2

        # create gammas [e_buy, e_sell, pi_buy, pi_sell]
        gammas = self.addVar("gamma", dim1, dim2, dim3, dim4, ub=1)

        # create approximated function values
        self.obj_pwl_f = self.addVar("f", dim1, dim2, dim3)  # [4, I, HL]

        # add constraints to the model
        self.model.addConstrs(
            (self.obj_pwl_f[d1, d2, d3] == np.sum(gammas[d1, d2, d3, :] * f_values[d1, d2, d3, :])
             for d1, d2, d3 in np.ndindex(dim1, dim2, dim3)), name="f"
        )
        self.model.addConstrs(
            (self.y[d1, d2, d3] == np.sum(gammas[d1, d2, d3, :] * y_values[d1, d2, d3, :])
             for d1, d2, d3 in np.ndindex(dim1, dim2, dim3)), name="y"
        )
        for d1, d2, d3 in np.ndindex(dim1, dim2, dim3):
            self.model.addSOS(GRB.SOS_TYPE2, gammas[d1, d2, d3, :])

        self.model.update()"""
