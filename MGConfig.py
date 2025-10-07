

class MG_config:
    def __init__(self, data):
        self.fsrr = data['fsrr']  # financial support rate
        self.usrr = data['usrr']  # utility support rate
        self.C_t_min = data['C_t_min']
        self.C_t_max = data['C_t_max']
        self.id = data['id']
        self.N = data['N']
        self.T = data['T']
        self.S = data['n_scen']
        self.probs = data['probs']
        self.load = data['l_s']
        self.pv_capacity = data['pv_capacity']
        self.dg_capacity = data['dg_capacity']
        self.es_capacity = data['es_capacity']
        self.es_charge = data['es_c']
        self.es_discha = data['es_d']
        self.pv_hourly = data['pv_hourly']
        self.pay_max = data['pay_max']
        self.pay_min = data['pay_min']
        self.dg_effi = data['dg_ef']
        self.alpha = data['alpha']
        self.theta_r = 0.75
        self.theta_c = 0.25
        self.mg_id = data['id']
        self.sv = data['sv']
        self.e_load = (1 + data['wtp']) * data['e_grid']
        self.tau = 0.9
        #self.tau_cutoff = self.tau + self.sv * (1 - self.tau)
        self.eta_r_Non = None
        self.eta_c_Non = None
        self.M = 1000
        self.es_cost = data['es_cost']
        self.dg_cost = data['dg_cost']
        self.shed_penalty = data['lsp']
        self.utility_cost = data['u_cost']

        self.n_index = range(self.N)
        self.t_index = range(self.T)
        self.s_index = range(self.S)

