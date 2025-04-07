""" This file is ADMM integrated as an environment with DRL"""
import copy
import math
from Functions import *
from EarlyStopping import EarlyStopping
import gym
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
import matplotlib.pyplot as plt




@dataclass
class ADMM_config:
    prim_eps: float = 0.05
    dual_eps: float = 0.05
    rand_rho: bool = False
    rho_init: float = 1
    xi: float = 1e-5
    mu: float = 10.
    tau_incr: float = 2.
    tau_decr: float = 2.

    def __init__(self, I, HL):
        self.yhat_init =  100 * np.ones((4, I, I, HL))
        self.z1_init =  np.zeros((I, HL))
        self.z2_init =  np.zeros((I, HL))

        self.l_yhat_init = np.zeros((4, I, I, HL))
        self.l_z1_init = np.zeros((I, HL))
        self.l_z2_init = np.zeros((I, HL))


class ADMM_env:
    def __init__(self, f, g, config: ADMM_config):
        # Sub-problems f: f(x, y), g: g(yhat)
        self.f = f
        self.g = g
        # Store parameters
        self.config = config
        # Reset History
        self.info = self.initialize_info()

    def step(self):
        """
        :param info: yhat, z1, z2, l_yhat, l_z1, l_z2, rho_yhat, rho_z1, rho_z2
        :return: info
        """

        # Solve f
        y_new, x_new = solve_f(f=self.f, info=self.info, xi=self.config.xi)

        # Box projection of x_new: the same shape
        z1_new = project_to_box(x=x_new)

        # Sphere projection of x_new: the same shape
        z2_new = project_to_sphere(x=x_new)

        # Solve g
        yhat_new = self.g.solve_with(y=y_new, info=self.info, xi=self.config.xi)

        # Update dual variables
        l_yhat_new = update_grad_ascent(info=self.info, keyword='yhat', grad=y_new - yhat_new)
        l_z1_new = update_grad_ascent(info=self.info, keyword='z1', grad=x_new - z1_new)
        l_z2_new = update_grad_ascent(info=self.info, keyword='z2', grad=x_new - z2_new)

        # Get residuals
        stacked_prim_vars = np.concatenate((y_new.flatten(), x_new.flatten(), x_new.flatten()))
        stacked_auxi_vars = np.concatenate((yhat_new.flatten(), z1_new.flatten(), z2_new.flatten()))
        stacked_auxi_vars_rhoed = np.concatenate((self.info['rho_yhat'] * yhat_new.flatten(),
                                            self.info['rho_z1'] * z1_new.flatten(),
                                            self.info['rho_z2'] * z2_new.flatten()))
        stacked_auxi_vars_rhoed_old = np.concatenate((self.info['rho_yhat'] * self.info['yhat'].flatten(),
                                          self.info['rho_z1'] * self.info['z1'].flatten(),
                                          self.info['rho_z2'] * self.info['z2'].flatten()))

        r_p_l2 = get_prim_residual(prim_vars=stacked_prim_vars, auxi_vars=stacked_auxi_vars)
        r_d_l2 = get_dual_residual(auxi_vars=stacked_auxi_vars_rhoed, auxi_vars_old=stacked_auxi_vars_rhoed_old)

        # Check convergence
        done = check_convergence(prim_eps=self.config.prim_eps, dual_eps=self.config.dual_eps,
                                 prim_res=r_p_l2, dual_res=r_d_l2, x=x_new)



        # Update info: new vars
        self.info = update_info(self.info, yhat_new, z1_new, z2_new, l_yhat_new, l_z1_new, l_z2_new)

        return done, self.info, r_p_l2, r_d_l2, x_new

    def reset(self):
        # Reset History
        self.info = self.initialize_info()

    def update_rhos(self):
        pass

    def initialize_info(self):
        info = {key: None for key in ['yhat', 'z1', 'z2',
                                      'l_yhat', 'l_z1', 'l_z2',
                                      'rho_yhat', 'rho_z1', 'rho_z2']}
        info['yhat'] = self.config.yhat_init
        info['z1'] = self.config.z1_init
        info['z2'] = self.config.z2_init

        info['l_yhat'] = self.config.l_yhat_init
        info['l_z1'] = self.config.l_z1_init
        info['l_z2'] = self.config.l_z2_init

        info['rho_yhat'] = self.config.rho_init
        info['rho_z1'] = self.config.rho_init
        info['rho_z2'] = self.config.rho_init

        return info

    @staticmethod
    def update_info(info: dict, yhat_new, z1_new, z2_new, l_yhat_new, l_z1_new, l_z2_new):
        info['yhat'] = yhat_new
        info['z1'] = z1_new
        info['z2'] = z2_new

        info['l_yhat'] = l_yhat_new
        info['l_z1'] = l_z1_new
        info['l_z2'] = l_z2_new
        return info


class RL_ADMM_Env(gym.Env):
    def __init__(self, env: ADMM_env):
        super(RL_ADMM_Env, self).__init__()

        # ADMM Env
        self.env = env

        # Action space
        self.action_space = gym.spaces.Box(low=env.config.action_min, high=env.config.action_max, dtype=np.float32)
        # Env initialization
        self.state = None
        self.state_dim = None
        self.state_feats = None
        self.vars = None
        self.dual_vars = None
        self.rhos = None
        self.reset()


    def step(self, action):
        # Build info

        '''# ADMM itr
        self.env.step(info)

        # Compute reward
        reward = get_RTI_reward(r_p=r_p_l2, r_d=r_d_l2,
                                r_p_past=self.state_feats['r_p_l2'],
                                r_d_past=self.state_feats['r_d_l2'],
                                weights=self.config.trend_weights)

        # Update state
        new_state = self.transition(rhos=rho_new, r_p_l2=r_p_l2, r_d_l2=r_d_l2)

        # Update state
        self.state = new_state

        # Update environment parameters
        self.update_repo(yhat=yhat_new, l=l_new, rho=rho_new, r_p_l2=r_p_l2, r_d_l2=r_d_l2)

        return new_state, reward, done, self.repo'''

    def reset(self, seed=None, options=None):  # required for GYM reset methods
        r_trend = np.full(self.config.trend_period, 1e4)

        # Build initial repository
        self.build_history()

        # Build state
        self.state_feats = {
            'rhos': self.rhos.values(),
            'r_p_l2': r_trend,
            'r_d_l2': r_trend}

        self.state = flatten_dictionary(self.state_feats)
        self.state_dim = self.state.size

    def transition(self, r_p_l2, r_d_l2):
        self.state_feats['r_p_l2'] = np.roll(self.state_feats['r_p_l2'], -1)
        self.state_feats['r_p_l2'][-1] = r_p_l2

        self.state_feats['r_d_l2'] = np.roll(self.state_feats['r_d_l2'], -1)
        self.state_feats['r_d_l2'][-1] = r_d_l2

        self.state_feats['rhos'] = self.rhos.values()

        return flatten_dictionary(self.state_feats)

    def build_history(self):
        self.vars = {
            'yhat': copy.copy(self.config.yhat_init),
            'z1': copy.copy(self.config.z1_init),
            'z2': copy.copy(self.config.z2_init)
        }
        self.dual_vars = {
            'yhat': copy.copy(self.config.l_yhat_init),
            'z1': copy.copy(self.config.l_z1_init),
            'z2': copy.copy(self.config.l_z2_init)
        }
        self.rhos = {
            'yhat': self.initialize_rho(),
            'z1': self.initialize_rho(),
            'z2': self.initialize_rho()
        }


    def update_history(self, yhat, z1, z2, l_yhat, l_z1, l_z2, rho, r_p_l2, r_d_l2):
        self.repo['yhat'] = yhat
        self.repo['z1'] = z1
        self.repo['z2'] = z2
        self.repo['l_yhat'] = l_yhat
        self.repo['l_z1'] = l_z1
        self.repo['l_z2'] = l_z2
        self.repo['rho'] = rho
        self.repo['r_p_l2'] = r_p_l2
        self.repo['r_d_l2'] = r_d_l2

    def initialize_rho(self):
        return self.action_space.sample() if self.config.rand_rho else self.config.rho_init

    def update_rhos(self, rhos_new):
        self.rhos['yhat'] = rhos_new[0]
        self.rhos['z1'] = rhos_new[1]
        self.rhos['z2'] = rhos_new[2]


def ADMM_run(admm_env: ADMM_env, max_itr):
    # Algorithm params
    itr = 1
    done = False

    # Info to save
    r_p_l2_hist, r_d_l2_hist = [], []

    # Algorithm
    with tqdm(initial=1, total=max_itr, desc='Progress: ') as pbar:
        while not done and itr <= max_itr:
            done, info, r_p_l2, r_d_l2, x = admm_env.step()
            r_p_l2_hist.append(r_p_l2)
            r_d_l2_hist.append(r_d_l2)

            # Update rhos
            if r_p_l2 >= admm_env.config.mu * r_d_l2:
                rho_coef = admm_env.config.tau_incr
            elif r_d_l2 >= admm_env.config.mu * r_p_l2:
                rho_coef = 1 / admm_env.config.tau_decr
            else:
                rho_coef = 1
            info['rho_yhat'] *= rho_coef
            info['rho_z1'] *= rho_coef
            info['rho_z2'] *= rho_coef

            # Iterate
            itr += 1
            pbar.update(1)

    # Plot
    plot_prim_dual_res(r_p_l2_hist, r_d_l2_hist)

    # Return
    return info, x

