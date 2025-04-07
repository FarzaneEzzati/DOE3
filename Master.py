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


class Master:
    def __init__(self, data, PWL=False):
        self.I = data['n_mgs']
        self.HL = data['HL']
        self.fsrr = data['fsrr']
        self.usrr = data['usrr']
        self.TFS = data['TFS']
        self.TUS = data['TUS']
        self.u_cost = data['u_cost']

        self.model = gp.Model('HigherLevel')
        self.model.setParam('OutputFlag', 0)
        self.model.setParam("Threads", 1)
        self.model.setParam("TimeLimit", 30)
        self.model.setParam("MIPGap", 0.01)

        # auxiliary variables e_hat, pi_hat
        self.e_buy_hat = self.addVar(name='e_buy_hat', dim1=self.I, dim2=self.I, dim3=self.HL)
        self.e_sell_hat = self.addVar(name='e_sell_hat', dim1=self.I, dim2=self.I, dim3=self.HL)
        self.pi_buy_hat = self.addVar(name='pi_buy_hat', dim1=self.I, dim2=self.I, dim3=self.HL)
        self.pi_sell_hat = self.addVar(name='pi_sell_hat', dim1=self.I, dim2=self.I, dim3=self.HL)
        self.yhat = np.stack((self.e_buy_hat, self.e_sell_hat, self.pi_buy_hat, self.pi_sell_hat), axis=0)

        # constraints
        self.model.addConstrs(
            (self.e_buy_hat[i, j, t] - self.e_sell_hat[j, i, t] == 0 for i, j, t in np.ndindex(self.I, self.I, self.HL)),
            name='e_clear')
        self.model.addConstrs(
            (self.pi_buy_hat[i, j, t] - self.pi_sell_hat[j, i, t] == 0 for i, j, t in np.ndindex(self.I, self.I, self.HL)),
            name='pi_clear')

        self.model.addConstr(
            np.average(self.pi_buy_hat.sum(axis=2).sum(axis=1), weights=self.fsrr) <= self.TFS, name='TFS')
        self.model.addConstr(
            np.average(self.u_cost * (self.e_buy_hat + self.e_sell_hat).sum(axis=2).sum(axis=1), weights=self.usrr) <= self.TUS, name='TUS')
        self.model.update()

        # reformulate quadratic term as PWL
        if PWL:
            self.reform_as_PWL(n_segments=100, z_min=np.zeros_like(data['y_max']), z_max=data['y_max'])

        self.obj_lagrangian = 0
        self.obj_pwl_f = 0
        self.PWL = PWL


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

    def solve_with(self, y, info, xi):
        lag_y = 0.5 * info['rho_yhat'] * (y - (info['rho_yhat'] * self.yhat - info['l_yhat']) / info['rho_yhat']) ** 2
        lag_y_sum = lag_y.sum()

        self.obj_lagrangian = xi * lag_y_sum
        self.model.setObjective(self.obj_lagrangian, sense=GRB.MINIMIZE)
        self.model.update()


        # solve model
        try:
            self.model.optimize()
        except gp.GurobiError as e:
            print(f"Gurobi Error: {e}")

        # Return z is model optimal or timed out, o.w. interrupt
        if self.model.Status == GRB.OPTIMAL or self.model.SolCount > 0:
            return self.get_yhat_opt()
        elif self.model.Status == GRB.INFEASIBLE:
            print(f'master model infeasible. Infeasible constraints: ')
            self.show_infeasible_const()
            raise ValueError('Algorithm stopped.')

    def evaluate_Q_ill(self, rho, xi):
        """
        report Q ill condition if either happens:
        k(Q) <= 1e6, lambda_min(Q) >= 1e-6, TIME_LIMIT reaches
        otherwise, report optimal y
        """
        Q_mat = 0.5 * rho * xi * np.diag((1 / (self.y_max ** 2)).flatten())
        Q_cond_number = np.linalg.cond(Q_mat)
        Q_min_eigen = np.min(np.linalg.eigvals(Q_mat))
        ill_condition = Q_cond_number >= 1e6 and Q_min_eigen <= 1e-6
        return ill_condition

    def reform_as_PWL(self, n_segments, z_min, z_max):
        dim1, dim2, dim3, dim4, dim5 = 4, self.I, self.I, self.HL, n_segments

        # uniform breakpoints and f_values
        y_values = np.zeros((dim1, dim2, dim3, dim4, dim5))
        for d1, d2, d3, d4 in np.ndindex(dim1, dim2, dim3, dim4):
            y_values[d1, d2, d3, d4, :] = np.linspace(z_min[d1, d2, d3, d4], z_max[d1, d2, d3, d4], dim5)
        f_values = y_values ** 2

        # create gammas [e_buy, e_sell, pi_buy, pi_sell]
        gammas = self.addVar("gamma", dim1, dim2, dim3, dim4, dim5, ub=1)

        # create approximated function values
        self.obj_pwl_f = self.addVar("f", dim1, dim2, dim3, dim4)  # [4, I, HL]

        # add constraints to the model
        self.model.addConstrs(
            (self.obj_pwl_f[d1, d2, d3, d4] == np.sum(gammas[d1, d2, d3, d4, :] * f_values[d1, d2, d3, d4, :])
             for d1, d2, d3, d4 in np.ndindex(dim1, dim2, dim3, dim4)), name="f"
        )
        self.model.addConstrs(
            (self.z[d1, d2, d3, d4] == np.sum(gammas[d1, d2, d3, d4, :] * y_values[d1, d2, d3, d4, :])
             for d1, d2, d3, d4 in np.ndindex(dim1, dim2, dim3, dim4)), name="z"
        )
        for d1, d2, d3, d4 in np.ndindex(dim1, dim2, dim3, dim4):
            self.model.addSOS(GRB.SOS_TYPE2, gammas[d1, d2, d3, d4, :])

        self.model.update()

    def get_yhat_opt(self):
        return np.array([var.X for var in self.yhat.flatten()]).reshape(self.yhat.shape)

    def show_infeasible_const(self):
        self.model.computeIIS()
        for c in self.model.getConstrs():
            if c.IISConstr:
                print('infeasible constr: ', c.ConstrName)