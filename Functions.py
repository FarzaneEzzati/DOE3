import math
import matplotlib.pyplot as plt
import numpy as np


# ADMM functions
def flatten_dictionary(dictionary):
        array_form = np.array([np.array(x).flatten() for x in dictionary.values()])
        return np.concatenate(array_form)


def update_grad_ascent(old_var, step_size, grad):
    return old_var + step_size * grad


def save_solutions(f, z, l, rho, xi, identifier):
    for i, f in enumerate(f):
        f.save_solutions(
            final_l=l[:, i, :, :],
            final_z=z[:, i, :, :],
            final_rho=rho,
            xi=xi,
            identifier=identifier)



def check_convergence(prim_eps, dual_eps, prim_res, dual_res):
    return prim_res <= prim_eps and dual_res <= dual_eps


def plot_l2_norm_pd(l2_norm_p_hist, l2_norm_d_hist, name):
    _, axs = plt.subplots(2, 1, figsize=(6, 5))
    axs[0].plot(l2_norm_p_hist, label='primal residual', color='red')
    axs[0].plot(l2_norm_d_hist, label='dual residual', color='blue')
    axs[0].legend()
    axs[1].set_ylim([0, 100])

    axs[1].plot(l2_norm_p_hist, label='primal residual', color='red')
    axs[1].plot(l2_norm_d_hist, label='dual residual', color='blue')
    axs[1].set_ylim([-0.1, 1])
    axs[1].set_yticks(np.linspace(0, 1, 6))
    axs[1].legend()

    plt.xlabel('Iteration')
    plt.ylabel('Residual (L2 Norm)')
    plt.savefig(f'Performance/{name}.jpg', dpi=300)
    plt.close()





def plot_rhos(rhos, name):
    rhos = np.array(rhos)
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(rhos[:, 0], label='y_hat', color='red')
    axs[1].plot(rhos[:, 1], label='z1')
    axs[2].plot(rhos[:, 2], label='z2', color='green')
    for ax in axs:
        ax.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.savefig(f'Performance/{name}.jpg', dpi=300)
    plt.close()