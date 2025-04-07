import math
import matplotlib.pyplot as plt
import numpy as np
from ADMM import ADMM_env, ADMM_config


# ADMM functions
def flatten_dictionary(dictionary):
        array_form = np.array([np.array(x).flatten() for x in dictionary.values()])
        return np.concatenate(array_form)


def get_prim_residual(prim_vars, auxi_vars):
    r_p = prim_vars - auxi_vars
    r_p_l2 = np.sqrt(np.sum(r_p ** 2))
    return r_p_l2


def get_dual_residual(auxi_vars, auxi_vars_old):
    r_d = auxi_vars - auxi_vars_old
    r_d_l2 = np.sqrt(np.sum(r_d ** 2))
    return r_d_l2


def update_grad_ascent(info, keyword, grad):
    old_var = info[f'l_{keyword}']
    step_size = info[f'rho_{keyword}']
    return old_var + step_size * grad


def save_solutions(f, z, l, rho, xi, identifier):
    for i, f in enumerate(f):
        f.save_solutions(
            final_l=l[:, i, :, :],
            final_z=z[:, i, :, :],
            final_rho=rho,
            xi=xi,
            identifier=identifier)


def get_RTI_reward(r_p, r_d, r_p_past, r_d_past, weights):
    r_p_tilde = np.average(r_p_past, weights=weights)
    r_d_tilde = np.average(r_d_past, weights=weights)

    reward_prim = max(-1, (r_p_tilde - r_p) / (r_p_tilde + 1e-5))
    reward_dual = max(-1, (r_d_tilde - r_d) / (r_d_tilde + 1e-5))
    return reward_prim + reward_dual


def solve_f(f, info, xi):
    y_new, x_new = [], []
    for i, fi in enumerate(f):
        yhat = info['yhat'][:, i, :, :]
        z1 = info['z1'][i, :]
        z2 = info['z2'][i, :]

        l_yhat = info['l_yhat'][:, i, :, :]
        l_z1 = info['l_z1'][i, :]
        l_z2 = info['l_z2'][i, :]

        rho_yhat = info['rho_yhat']
        rho_z1 = info['rho_z1']
        rho_z2 = info['rho_z2']

        y, x = fi.solve_with(yhat=yhat, z1=z1, z2=z2,
                          l_yhat=l_yhat, l_z1=l_z1, l_z2=l_z2,
                          rho_yhat=rho_yhat, rho_z1=rho_z1, rho_z2=rho_z2,
                          xi=xi)
        y_new.append(y)
        x_new.append(x)
    return np.stack(y_new, axis=1), np.array(x_new)


def solve_g(g, info, xi):
    return g.solve_with(y=info['yhat'], l_yhat=info['l_yhat'], rho_yhat=info['rho_yhat'], xi=xi)


def project_to_box(x):
    return np.clip(x, 0, 1)


def project_to_sphere(x):
    x_flat = x.flatten()
    x_norm = np.linalg.norm(x)
    n = x_flat.size
    unit_vector = np.ones_like(x_flat)
    x_projected = 0.5 * math.sqrt(n) * (x_flat / x_norm) + 0.5 * unit_vector
    return np.reshape(x_projected, x.shape)


def check_convergence(prim_eps, dual_eps, prim_res, dual_res, x):
    residual_cond = all((prim_res <= prim_eps, dual_res <= dual_eps))
    binary_cond = np.all(np.isclose(x, 0, atol=1e-4) | np.isclose(x, 1, atol=1e-4))
    return residual_cond and binary_cond


def update_info_rhos(rhos, info: dict):
    info['rho_yhat'] = rhos['yhat']
    info['rho_z1'] = rhos['z1']
    info['rho_z2'] = rhos['z2']
    return info


def update_info(info: dict, yhat_new, z1_new, z2_new, l_yhat_new, l_z1_new, l_z2_new):
    info['yhat'] = yhat_new
    info['z1'] = z1_new
    info['z2'] = z2_new

    info['l_yhat'] = l_yhat_new
    info['l_z1'] = l_z1_new
    info['l_z2'] = l_z2_new
    return info


def modify_rhos(rhos, r_p_l2, r_d_l2):
    pass


def plot_prim_dual_res(r_p_l2_hist, r_d_l2_hist):
    _, axs = plt.subplots(2, 1, figsize=(5, 5))
    axs[0].plot(r_p_l2_hist, label='primal', color='red')
    axs[0].plot(r_d_l2_hist, label='dual', color='blue')
    axs[0].legend()

    axs[1].plot(r_p_l2_hist, label='primal', color='red')
    axs[1].plot(r_d_l2_hist, label='dual', color='blue')
    axs[1].set_ylim([0, 1])
    axs[1].legend()

    plt.xlabel('Iteration')
    plt.ylabel('Residual (L2 Norm)')



