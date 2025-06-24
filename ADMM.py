""" This file is ADMM integrated as an environment with DRL"""
import copy
from Functions import *
import gym
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


############################## Configuration #########################
@dataclass
class ADMM_config:
    yhat_epsilon: float = 0.01
    z_epsilon: float = 0.01

    def __init__(self, y_shape, x_shape, is_lp_box):
        self.is_lp_box = is_lp_box
        self.yhat_init = 100 * np.ones(y_shape)
        self.l_yhat_init = np.zeros(y_shape)

        if is_lp_box:
            self.z2_init = np.zeros(x_shape)
            self.l_z2_init = np.zeros(x_shape)


############################## ADMM ENV #########################
class ADMM_env:
    def __init__(self, f, g, config: ADMM_config):
        # Sub-problems f: f(x, y), g: g(yhat)
        self.f = f
        self.g = g
        # Store parameters
        self.config = config
        # Reset History
        self.info = self.initialize_info()
        if self.config.is_lp_box:
            self.num_rhos = 2
            self.center = 0.5 * np.ones_like(self.info['z2'].shape[1])
            r = np.sqrt(self.center.size)/2
            self.radius = r
        else:
            self.num_rhos = 1

    def step(self, rhos, gamma0, gamma1, eta):
        yhat = self.info['yhat']
        l_yhat = self.info['l_yhat']
        rho_yhat = rhos[0]

        # Solve f
        y_new, x_new = self.solve_f(rhos=rhos)

        # Solve g
        yhat_new = self.g.solve_with(y=y_new, rho_yhat=rho_yhat, l_yhat=l_yhat)

        # Primal and Dual residuals for yhat
        p_r_2_yhat = np.linalg.norm(y_new - yhat_new)
        d_r_2_yhat = np.linalg.norm(rho_yhat * (yhat_new - yhat))

        # Update dual variables for yhat
        l_yhat_new = l_yhat + rho_yhat * (y_new - yhat_new)

        # Update yhat and x related info
        self.update_info(yhat_new=yhat_new, l_yhat_new=l_yhat_new, x_new=x_new)

        # Project and update if lp_box
        if self.config.is_lp_box:
            z1 = self.info['z1']
            z2 = self.info['z2']
            l_z1 = self.info['l_z1']
            l_z2 = self.info['l_z2']
            rho_z1 = rhos[1]
            rho_z2 = rhos[2]

            # Update z1
            z1_new = self.info['z1']

            ### Update z2
            z2_new = self.update_z2(z2=z2, l_z2=l_z2, rho_z2=rho_z2,
                                    x=x_new, gamma0=gamma0, gamma1=gamma1,
                                    eta=eta)

            # Update dual variables
            l_z1_new = l_z1 + rho_z1 * (x_new - z1_new)
            l_z2_new = l_z2 + rho_z2 * (x_new - z2_new)

            # Squared norm of residuals
            p_r_2_z1 = np.linalg.norm(x_new - z1_new)
            d_r_2_z1 =  np.linalg.norm(rho_z1 * (z1_new - z1))
            p_r_2_z2 =  np.linalg.norm(x_new - z2_new)
            d_r_2_z2 =  np.linalg.norm(rho_z2 * (z2_new - z2))

            # Update z1 and z2 related info
            self.update_info_lp_box(z1_new=z1_new, z2_new=z2_new, l_z1_new=l_z1_new, l_z2_new=l_z2_new)
        else:
            p_r_2_z1, d_r_2_z1 =  0, 0
            p_r_2_z2, d_r_2_z2 = 0, 0

        # Check convergence
        r_yhat_l2 = np.sqrt(p_r_2_yhat + d_r_2_yhat)
        r_z1_l2 = np.sqrt(p_r_2_z1 + d_r_2_z1)
        r_z2_l2 = np.sqrt(p_r_2_z2 + d_r_2_z2)
        done = all([
            r_yhat_l2 <= self.config.yhat_epsilon,
            r_z1_l2 <= self.config.z_epsilon,
            r_z2_l2 <= self.config.z_epsilon
        ])

        # Separated residuals
        p_r_separated = [np.sqrt(p_r_2_yhat), np.sqrt(p_r_2_z1), np.sqrt(p_r_2_z2)]
        d_r_separated = [np.sqrt(d_r_2_yhat), np.sqrt(d_r_2_z1), np.sqrt(d_r_2_z2)]
        r_separated = [p_r_separated, d_r_separated]


        return done, r_separated, self.info, x_new

    def reset(self):
        self.info = self.initialize_info()

    def update_z1(self, z1, l_z1, rho_z1, x, gamma0, gamma1, eta):
        lagrang_G = -l_z1 - rho_z1 * (x - z1)
        negative_G = gamma0 * np.minimum(2 * z1, 0)
        positive_G = gamma1 * np.maximum(2 * (z1 - 1), 0)
        G = lagrang_G + negative_G + positive_G
        z1_new = z1 - eta * G
        return z1_new

    def update_z2(self, z2, l_z2, rho_z2, x, gamma0, gamma1, eta):
        # Riemannain gradient
        lagrang_G = -l_z2 - rho_z2 * (x - z2)
        negative_G = gamma0 * np.minimum(2 * z2, 0)
        positive_G = gamma1 * np.maximum(2 * (z2 - 1), 0)
        G = lagrang_G + negative_G + positive_G
        v = z2 - self.center
        r2 = self.radius ** 2
        vvT = np.array([vi[:, None] @ vi[:, None].T for vi in v])
        vvT_unit = vvT / r2
        vvTG = np.array([vvTi @ gi for gi, vvTi in zip(G, vvT_unit)])
        grad_G = G - vvTG

        z2_new = z2 - eta * grad_G
        z2_new = self.project_to_surface(z2_new)
        return z2_new

    def initialize_info(self):
        info = dict({})
        info['yhat'] = self.config.yhat_init
        info['l_yhat'] = self.config.l_yhat_init
        info['x'] = None

        if self.config.is_lp_box:
            info['z1'] = self.config.z1_init
            info['l_z1'] = self.config.l_z1_init
            info['z2'] = self.config.z2_init
            info['l_z2'] = self.config.l_z2_init

        return info

    def project_to_surface(self, var):
        v = var - self.center
        v_norm = np.linalg.norm(v, axis=1)
        v_unit = np.array([nom / denom for nom, denom in zip(v, v_norm)])
        var_new = self.center + self.radius * v_unit
        return var_new

    def update_info(self, yhat_new, l_yhat_new,  x_new):
        self.info['yhat'] = yhat_new
        self.info['l_yhat'] = l_yhat_new
        self.info['x'] = x_new

    def update_info_lp_box(self, z1_new, l_z1_new, z2_new, l_z2_new):
        self.info['z1'] = z1_new
        self.info['l_z1'] = l_z1_new
        self.info['z2'] = z2_new
        self.info['l_z2'] = l_z2_new

    def solve_f(self, rhos):
        y_new, x_new = [], []

        for i, fi in enumerate(self.f):
            kwargs = {
                'yhat': self.info['yhat'][:, i, :, :],
                'l_yhat': self.info['l_yhat'][:, i, :, :],
                'rho_yhat': rhos[0],
            }

            if self.config.is_lp_box:
                kwargs.update({
                    'z1': self.info['z1'][i, :],
                    'l_z1': self.info['l_z1'][i, :],
                    'rho_z1': rhos[1],
                    'z2': self.info['z2'][i, :],
                    'l_z2': self.info['l_z2'][i, :],
                    'rho_z2': rhos[2],
                })
            y, x = fi.solve_with(**kwargs)
            y_new.append(y)
            x_new.append(x)

        y_new = np.transpose(np.stack(y_new, axis=0), (1, 0, 2, 3))
        x_new = np.stack(x_new, axis=0)

        return y_new, x_new


############################## Run ADMM Config #########################
@dataclass
class ADMM_run_config:
    # For run itself
    mu_yhat: float = 10
    mu_z1: float = 10
    mu_z2: float = 10
    tau_incr: float = 2
    tau_decr: float = 2

    # For env
    rho_yhat: float = 1e-9
    rho_z1: float = 10
    rho_z2: float = 10
    gamma0: float = 100.0
    gamma1: float = 100.0
    eta: float = 1e-2


############################## Run ADMM #########################
def ADMM_run(admm_env: ADMM_env, config: ADMM_run_config, name_to_save, max_itr):
    # env params
    kwargs = {
        'rhos': [config.rho_yhat, config.rho_z1, config.rho_z2],
        'gamma0': config.gamma0,
        'gamma1': config.gamma1,
        'eta': config.eta}

    # Info to save
    x_hist = []
    r_p_separated_hist = {key: [] for key in ['yhat', 'z1', 'z2']}
    r_d_separated_hist = {key: [] for key in ['yhat', 'z1', 'z2']}
    rhos_hist = []  # rho_yhat, rho_z1, rho_z2

    # Reset env
    admm_env.reset()

    # Algorithm
    itr = 1
    done = False
    with tqdm(initial=0, total=max_itr, desc='Progress: ') as pbar:
        while not done and itr <= max_itr:
            done, r_separated, info, x = admm_env.step(**kwargs)
            r_p_separated, r_d_separated = r_separated[0], r_separated[1]

            r_p_separated_hist['yhat'].append(r_p_separated[0])
            r_p_separated_hist['z1'].append(r_p_separated[1])
            r_p_separated_hist['z2'].append(r_p_separated[2])

            r_d_separated_hist['yhat'].append(r_d_separated[0])
            r_d_separated_hist['z1'].append(r_d_separated[1])
            r_d_separated_hist['z2'].append(r_d_separated[2])
            x_hist.append(x)

            # Update rhos separately
            yhat_coef = (
                config.tau_incr if r_p_separated[0] >= config.mu_yhat * r_d_separated[0]
                else 1 / config.tau_decr if r_d_separated[0] >= config.mu_yhat * r_p_separated[0]
                else 1)
            z1_coef = (
                config.tau_incr if r_p_separated[1] >= config.mu_z1 * r_d_separated[1]
                else 1 / config.tau_decr if r_d_separated[1] >= config.mu_z2 * r_p_separated[1]
                else 1)
            z2_coef = (
                config.tau_incr if r_p_separated[2] >= config.mu_z2 * r_d_separated[2]
                else 1 / config.tau_decr if r_d_separated[2] >= config.mu_z2 * r_p_separated[2]
                else 1)
            kwargs['rhos'][0] *= yhat_coef
            kwargs['rhos'][1] *= z1_coef
            kwargs['rhos'][2] *= z2_coef
            kwargs['eta'] *= 0.97

            # Save rhos
            rhos_hist.append(kwargs['rhos'])

            # Iterate
            itr += 1
            pbar.update(1)

    plot_residuals(r_p_separated_hist, r_d_separated_hist, name_to_save+'sep_residual')
    # Return
    return info, kwargs['rhos'], x_hist


def plot_residuals(r_p_separated, r_d_separated, name):
    fig, axs = plt.subplots(2, 1, figsize=(6, 5))
    plt.subplots_adjust(hspace=0.6)
    colors = {'yhat': 'red', 'z2': 'blue'}
    for key, value in r_p_separated.items():
        if key != 'z1':
            axs[0].plot(value, label=key, color=colors[key])
            axs[0].set_title('Primal Residuals')
    for key, value in r_d_separated.items():
        if key != 'z1':
            axs[1].plot(value, label=key, color=colors[key])
            axs[1].set_title('Dual Residuals')

    for ax in axs:
        ax.legend()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual (L2 Norm)')

    plt.savefig(f'Performance/{name}.jpg', dpi=300)
    plt.close()

############################## RL ADMM Configuration #########################
@dataclass
class DRL_config:
    action_min: float = 0.00001
    action_max: float = 1000
    trend_period: int = 5
    trend_weights: np.ndarray = np.array([0.1, 0.1, 0.2, 0.3, 0.3])


############################## RL ADMM ENV #########################
class DRL_ADMM_env(gym.Env):
    def __init__(self, env: ADMM_env, config: DRL_config):
        super(DRL_ADMM_env, self).__init__()
        # ADMM Env
        self.env = env
        # Action
        self.action_dim = self.env.num_rhos
        self.action_space = gym.spaces.Box(low=config.action_min, high=config.action_max,
                                           shape=(self.action_dim,), dtype=np.float32)
        self.config = config
        # Env initialization
        self.state = None
        self.state_dim = None
        self.state_feats = None
        self.reset()

    def step(self, action):
        # ADMM itr
        done, r_p_l2, r_d_l2, info, _ = self.env.step(action)
        residuals = [r_p_l2, r_d_l2]

        # Compute reward
        reward = self.get_reward(r_p_l2=r_p_l2, r_d_l2=r_d_l2, done=done)

        # Update state
        new_state = self.transition(rhos=action, r_p_l2=r_p_l2, r_d_l2=r_d_l2)

        # Update state
        self.state = new_state

        return new_state, reward, done, residuals, info

    def reset(self, seed=None, options=None):
        # Initialize info in ADMM-env
        self.env.reset()
        init_rhos = self.action_space.sample()

        # Build state features and flatten
        self.state_feats = {
            'rho_yhat': init_rhos[0],
            'l2_norm_p': np.full(self.config.trend_period, 100),
            'l2_norm_d': np.full(self.config.trend_period, 100)}
        # Add state of lp_box if True
        if self.env.config.is_lp_box:
            self.state_feats['rho_z1'] = init_rhos[1],
            self.state_feats['rho_z2'] = init_rhos[2],

        self.state = flatten_dictionary(self.state_feats)
        self.state_dim = self.state.size

    def get_reward(self, r_p_l2, r_d_l2, done):
        r_tilde = np.average(self.state_feats['l2_norm_p'] + self.state_feats['l2_norm_p'], weights=self.config.trend_weights)
        r = r_p_l2 + r_d_l2
        r_compare = max(-100, (r_tilde - r)/r_tilde)
        r_converge = self.config.convergence_reward if done else 0
        return r_compare + r_converge

    def transition(self, rhos, r_p_l2, r_d_l2):
        self.state_feats['l2_norm_p'] = np.roll(self.state_feats['l2_norm_p'], -1)
        self.state_feats['l2_norm_p'][-1] = r_p_l2

        self.state_feats['l2_norm_d'] = np.roll(self.state_feats['l2_norm_d'], -1)
        self.state_feats['l2_norm_d'][-1] = r_d_l2

        self.state_feats['rho_yhat'] = rhos[0]

        if self.env.config.is_lp_box:
            self.state_feats['rho_z1'] = rhos[1]
            self.state_feats['rho_z2'] = rhos[2]

        return flatten_dictionary(self.state_feats)


