from brian2 import *

class Parameter:
    def __init__(self, value, unit=1):
        
        if type(value) is Quantity:
            print('ERROR: Give value without unit')
            
        if type(value) is bool:
            self.quantity = value
            
        else:
            self.quantity = value * unit
            
        self.unit = unit
        self.value = value
        self.used = False
        
      
    def get_param(self):
        '''Gets the value and unit of a parameter'''
        self.used = True
        return self.quantity
    
    
    def set_param(self, x):
        '''Modifies the value of a parameter'''
        self.quantity = x * self.unit
    
    
def get_used_params(param_group):
    '''Only parameters that have actually been used are carried around'''
    used_param_group = {}

    for param_name in param_group:

        if param_group[param_name].used:
            used_unit = param_group[param_name].unit
            used_val = param_group[param_name].quantity

            if type(param_group[param_name].quantity) is not str:
                used_val = used_val / used_unit

            used_param_group[param_name] = Parameter(used_val, used_unit)

    return used_param_group


def get_default_net_params():
    '''Contains the default set of parameters reported in the paper'''

    # Seed
    default_params = {'seed': Parameter(123)}
    
    # AdEx neuron, shared parameters
    default_params = {**default_params,
                      **{'v_stop': Parameter(30, mV), 'tau_refr': Parameter(3, ms), 'slope': Parameter(2.5, mV)}
                      }
    
    # AdEx neuron, population-specific parameters
    default_params = {**default_params,
                      **{'g_leak_t': Parameter(11, nS), 'v_thres_t': Parameter(-44, mV),'v_reset_t': Parameter(-46, mV),
                         'g_leak_a': Parameter(8, nS), 'v_thres_a': Parameter(-48, mV),'v_reset_a': Parameter(-42, mV), 
                         'g_leak_b': Parameter(6, nS), 'v_thres_b': Parameter(-40, mV),'v_reset_b': Parameter(-57, mV),
                         'g_leak_c': Parameter(5, nS), 'v_thres_c': Parameter(-40, mV),'v_reset_c': Parameter(-52, mV),
                         
                         'mem_cap_t': Parameter(200, pF), 'e_rever_t': Parameter(-70, mV),
                         'mem_cap_a': Parameter(200, pF), 'e_rever_a': Parameter(-60, mV),
                         'mem_cap_b': Parameter(100, pF), 'e_rever_b': Parameter(-55, mV),
                         'mem_cap_c': Parameter(100, pF), 'e_rever_c': Parameter(-57, mV)}
                      }
    
    # AdEx neuron, adaptation parameters:
    default_params = {**default_params,
                      **{'J_pot_t': Parameter(0, nS), 'J_spi_t': Parameter(150, pA), 'tau_adapt_t': Parameter(200, ms),
                         'J_pot_a': Parameter(4, nS), 'J_spi_a': Parameter(85, pA), 'tau_adapt_a': Parameter(200, ms),
                         'J_pot_b': Parameter(6, nS), 'J_spi_b': Parameter(25, pA), 'tau_adapt_b': Parameter(50, ms),
                         'J_pot_c': Parameter(2.5, nS), 'J_spi_c': Parameter(20, pA), 'tau_adapt_c': Parameter(100, ms),
                        }
                      }
    
    # Size and background current to neuronal populations:
    default_params = {**default_params,
                      **{'n_t': Parameter(5300), 'curr_bg_t': Parameter(285, pA),
                         'n_a': Parameter(2700), 'curr_bg_a': Parameter(140, pA),
                         'n_b': Parameter(150), 'curr_bg_b': Parameter(180, pA),
                         'n_c': Parameter(100), 'curr_bg_c': Parameter(160, pA)}
                      }

    # Connectivity parameters:
    default_params = {**default_params,
                      **{'prob_tt': Parameter(0.08), 'prob_at': Parameter(0.11), 'prob_bt': Parameter(0.20), 'prob_ct': Parameter(0.20),
                         'prob_ta': Parameter(0.04), 'prob_aa': Parameter(0.15), 'prob_ba': Parameter(0.20), 'prob_ca': Parameter(0.20),
                         'prob_tb': Parameter(0.20), 'prob_ab': Parameter(0.20), 'prob_bb': Parameter(0.20), 'prob_cb': Parameter(0.20),
                         'prob_tc': Parameter(0.20), 'prob_ac': Parameter(0.20), 'prob_bc': Parameter(0.20), 'prob_cc': Parameter(0.20)}
                      }

    # Weights (peak conductances):
    default_params = {**default_params,
                      **{'g_tt': Parameter(0.2, nS), 'g_at': Parameter(0.2, nS), 'g_bt': Parameter(0.5, nS), 'g_ct': Parameter(0.05, nS),
                         'g_ta': Parameter(0.2, nS), 'g_aa': Parameter(0.2, nS), 'g_ba': Parameter(0.7, nS), 'g_ca': Parameter(0.1, nS),
                         'g_tb': Parameter(0.8, nS), 'g_ab': Parameter(2.15, nS), 'g_bb': Parameter(6, nS), 'g_cb': Parameter(5, nS),
                         'g_tc': Parameter(15, nS), 'g_ac': Parameter(15, nS), 'g_bc': Parameter(9, nS), 'g_cc': Parameter(3, nS)}
                      }
    
    # Synaptic parameters:
    default_params = {**default_params,
                      **{'tau_d_e': Parameter(2, ms), 'tau_d_i': Parameter(4, ms), 'e_e': Parameter(0, mV), 'e_i': Parameter(-70, mV),
                         'tau_l': Parameter(1, ms)}
                      }
    
    # Extra parameters for synaptic depression and heterogeneity:
    default_params = {**default_params,
                      **{'tau_dep': Parameter(250, ms), 'eta': Parameter(0.18), 'std_e_rever': Parameter(0.05),
                         'std_g_leak': Parameter(0.25), 'std_cond': Parameter(0.25), 'std_delay': Parameter(0.25)}
                      }

    return default_params



def get_dft_test_params():
    '''Contains parameters for the detection and characterization of SPW events'''

    # Simulation parameters
    dft_test_params = {'seed': Parameter(123),
                       'sim_dt': Parameter(0.10, ms),
                       'prep_time': Parameter(2, second),
                       'sim_time': Parameter(10, second),
                       }

    # Parameters for the analysis of SPWs (explained in detail in "simulations")
    dft_test_params = {**dft_test_params,
                       **{'record_spikes': Parameter(False),
                          'check_second_peaks': Parameter(True),
                          'min_peak_height': Parameter(50, pA),
                          'min_peak_dist': Parameter(0.4, second),
                          'min_subpeak_dist': Parameter(0.05, second),
                          'lfp_cutoff': Parameter(10, Hz),
                          'gauss_window': Parameter(3, ms),
                          'spw_window': Parameter(0.2, second),
                          'baseline_start': Parameter(0.3, second),
                          'baseline_end': Parameter(0.2, second)}
                       }

    # Number of neurons sampled to record variables
    dft_test_params = {**dft_test_params,
                       **{'rec_adapt_num': Parameter(50),
                          'rec_mempo_num': Parameter(50),
                          'rec_curr_num': Parameter(50)}}

    return dft_test_params
    