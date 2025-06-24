""" This file is ADMM without integrating with DRL"""

import numpy as np
from time import time
import matplotlib.pyplot as plt


class EarlyStopping:
    def __init__(self, patience, min_delta, identifier):
        self.patience = patience
        self.min_delta = min_delta
        self.best_r_p_l2 = float('inf')
        self.best_r_d_l2 = float('inf')
        self.counter = 0
        self.save_path = f'Models/{identifier}.pt'

    def __call__(self, r_p_l2, r_d_l2):
        better_r = (r_p_l2 <= self.best_r_p_l2 - self.min_delta and r_d_l2 <= self.best_r_d_l2 - self.min_delta)
        if better_r:
            self.best_r_p_l2 = r_p_l2
            self.best_r_d_l2 = r_d_l2
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience


class ADMM:
    def __init__(self, fy_ip, gz_ip, args, fy_lp=None, gz_lp=None):
        self.fy_ip = fy_ip
        self.gz_ip = gz_ip

        self.fy_lp = fy_lp
        self.gz_lp = gz_lp

        self.l_init = args['l_init']
        self.rho_init = args['rho_init']
        self.z_init = args['z_init']
        self.xi = args['xi']
        self.prim_epsilon = args['prim_epsilon']
        self.dual_epsilon = args['dual_epsilon']
        self.mu = args['mu']
        self.tau_incr = args['tau_incr']
        self.tau_decr = args['tau_decr']
        self.max_itr = args['max_itr']
        self.update_rule = args['update']
        self.beta = args['beta']
        self.print_every = args['print_every']
        self.identifier = args['identifier']

        # Early stopping class
        self.early_stopping = EarlyStopping(
            patience=args['patience'],
            min_delta=args['min_delta'],
            identifier=self.identifier)

    def start(self):
        r_p_l2_hist, r_d_l2_hist = [], []
        z, l, rho, xi = self.z_init, self.l_init, self.rho_init, self.xi
        y_new_ip = None
        done = False
        itr = 0

        print('ADMM algorithm started.')
        start = time()
        while not done and itr < self.max_itr:
            # Solve fy_ip and gz_ip
            y_new_ip = self.solve_fy(z=z, l=l, rho=rho)
            z_new_ip = self.solve_gz(y=y_new_ip, l=l, rho=rho)

            if self.update_rule == 'classic':
                # Update lambda
                l_new = self.update_lambda_classic(y=y_new_ip, z=z_new_ip, l=l, rho=rho)
            elif 'hybrid':

                y_new_lp = self.solve_fy(z=z, l=l, rho=rho, ip=False)
                z_new_lp = self.solve_gz(y=y_new_lp, l=l, rho=rho, ip=False)
                l_new = self.update_lambda_hybrid(y_ip=y_new_ip, z_ip=z_new_ip,
                                                  y_lp=y_new_lp, z_lp=z_new_lp,
                                                  l=l, rho=rho)

            # Calculate residuals
            r_p_l2, r_d_l2 =  self.calculate_residuals(y_new=y_new_ip, z_new=z_new_ip, z_old=z, rho=rho)
            r_p_l2_hist.append(r_p_l2)
            r_d_l2_hist.append(r_d_l2)

            # Print results
            if (itr + 1) % self.print_every == 0:
                self.print_results(r_p_l2, r_d_l2, rho, itr, start)

            # Update rho
            rho_new = self.update_rho(rho, r_p_l2, r_d_l2)

            # Check convergence
            done = self.converged(r_p_l2, r_d_l2)

            # Update z, l, rho
            z, l, rho = z_new_ip, l_new, rho_new

            # Check early stopping
            stop = self.early_stopping(r_p_l2, r_d_l2)
            if stop or (itr + 1 == self.max_itr):
                # Print summary
                finish = time()
                print(f'ADMM Finished. Early-stop: {stop}. Duration: {(finish - start):0.2f}s, {(finish - start) / itr:0.2f}s/itr')

                # Plot residual
                self.plot_residual(r_p_l2_hist, r_d_l2_hist)

                # Store solutions
                self.save_solutions(z=z, l=l, rho=rho)

                # Terminate
                break

            else:
                # Increment itr
                itr += 1



    def update_rho(self, rho, r_p_l2, r_d_l2):
        if r_p_l2 >= self.mu * r_d_l2:
            rho *= self.tau_incr
        elif r_d_l2 >= self.mu * r_p_l2:
            rho /= self.tau_decr
        else:
            pass
        return rho

    @staticmethod
    def update_lambda_classic(y, z, l, rho):
        l = l + rho * (y - z)
        return l

    def update_lambda_hybrid(self, y_ip, z_ip, y_lp, z_lp, l, rho):
        beta = self.beta
        l = l + rho * (beta * (y_ip - z_ip) + (1-beta) * (y_lp - z_lp))
        return l

    def solve_fy(self, z, l, rho, ip=True):
        y_new = []
        F = self.fy_ip if ip else self.fy_lp
        for i, f in enumerate(F):
            y = f.solve_with(z=z[:, i, :, :], l=l[:, i, :, :], rho=rho, xi=self.xi)
            y_new.append(y)
        return np.stack(y_new, axis=1)

    def solve_gz(self, y, l, rho, ip=True):
        if ip:
            return self.gz_ip.solve_with(y=y, rho=rho, l=l, xi=self.xi)
        else:
            return self.gz_lp.solve_with(y=y, rho=rho, l=l, xi=self.xi)

    @staticmethod
    def calculate_residuals(y_new, z_new, z_old, rho):
        r_p = y_new - z_new
        r_d = rho * (z_new - z_old)
        r_p_l2 = np.sqrt(np.sum(r_p ** 2))
        r_d_l2 = np.sqrt(np.sum(r_d ** 2))
        return r_p_l2, r_d_l2

    def converged(self, r_p_l2, r_d_l2):
        done = (r_p_l2 <= self.prim_epsilon) and (r_d_l2 <= self.dual_epsilon)
        if done:
            print(f'Algorithm converged with primal residual: {r_p_l2:0.6f} and dual residual: {r_d_l2:0.6f}')
        return done

    def print_results(self, r_p_l2, r_d_l2, rho, itr, start):
        print(f'Itr: {itr + 1}/{self.max_itr}  ' +
              f'r_p_l2: {r_p_l2:0.2e}   ' +
              f'r-d_l2: {r_d_l2:0.2e}   ' +
              f'rho: {rho:0.2e}   ' +
              f'{(itr + 1) /(time()-start):0.2f}itr/s')

    def plot_residual(self, r_p_l2, r_d_l2):
        fig, axs = plt.subplots(2, 1, figsize=(8, 7))
        for ax in axs:
            ax.plot(r_p_l2, label='primal')
            ax.plot(r_d_l2, label='dual')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Residual L2 Norm')
            ax.legend()
        axs[1].set_ylim([0, 1])
        plt.savefig(f'Performance/{self.identifier}residual_hist.jpg', dpi=300, bbox_inches='tight')

    def save_solutions(self, z, l, rho):
        for i, f in enumerate(self.fy_ip):
            f.save_solutions(final_l=l[:, i, :, :],
                               final_z=z[:, i, :, :],
                               final_rho=rho,
                               xi=self.xi,
                               identifier=self.identifier)
