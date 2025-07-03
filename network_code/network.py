from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from network_code.utils import Connectivity
from network_code.parameters import *


def build_network(net_params, plastic_ee=False, heterogenous=False):
    '''Builds the spiking network in Brian'''
    
    # Sets the seed for the random variables
    net_seed = int(net_params['seed'].get_param())
    seed(net_seed)

    # Synaptic depression is implemented at the neuron level, since it only depends on the pre-synaptic neuron
    neuron_eqs_dep = '''
            deff / dt = (1 - eff) / tau_dep : 1
            tau_dep : second
            eta : 1
        '''
    
    # Equations of the AdEx model
    neuron_eqs_common = '''
        dv/dt = (g_leak*(e_rever - v) + g_leak*slope*exp((v-v_thres)/slope) + curr_syn + curr_bg - curr_adapt)/mem_cap : volt (unless refractory)
        dcurr_adapt/dt = (-curr_adapt + J_pot*(v-e_rever))/tau_adapt : amp
        curr_syn = curr_t + curr_a + curr_b + curr_c : amp
        mem_cap : farad
        g_leak : siemens
        e_rever : volt
        v_reset : volt
        v_thres : volt
        v_stop : volt
        tau_refr : second
        curr_bg : amp
        slope : volt
        J_pot : siemens
        tau_adapt : second
        J_spi : amp
        g_it : siemens
        g_ia : siemens
        g_ib : siemens
        g_ic : siemens
        '''

    # Conductance-based synapses
    curr_eqs = '''
        curr_t = g_t * (e_e - v) : amp
        curr_a = g_a * (e_e - v): amp
        curr_b = g_b * (e_i - v): amp
        curr_c = g_c * (e_i - v): amp
        dg_t / dt = -g_t / tau_d_e : siemens
        dg_a / dt = -g_a / tau_d_e: siemens
        dg_b / dt = -g_b / tau_d_i: siemens
        dg_c / dt = -g_c / tau_d_i: siemens
        e_e : volt
        e_i : volt
        tau_d_e: second
        tau_d_i : second
    '''
    
    all_eqs = neuron_eqs_common + curr_eqs
    if plastic_ee:
        all_eqs_exc = neuron_eqs_dep + neuron_eqs_common + curr_eqs

    n_t = int(net_params['n_t'].get_param())
    n_a = int(net_params['n_a'].get_param())
    n_b = int(net_params['n_b'].get_param())
    n_c = int(net_params['n_c'].get_param())

    # Equations for the depressing model include, for each spike, a reduction of synapitc efficacy (for the E populations)
    if plastic_ee:
        pop_t = NeuronGroup(n_t, model=all_eqs_exc, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            eff = clip(eff*(1 - eta), 0, 1)
                            ''', refractory='tau_refr', method='euler', name='pop_t')
    
        pop_a = NeuronGroup(n_a, model=all_eqs_exc, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            eff = clip(eff*(1 - eta), 0, 1)
                            ''', refractory='tau_refr', method='euler', name='pop_a')
        
        pop_b = NeuronGroup(n_b, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_b')
        
        pop_c = NeuronGroup(n_c, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_c')

    # Standard equations include an adaptation increment happening together with the voltage reset
    else:
        pop_t = NeuronGroup(n_t, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_t')
    
        pop_a = NeuronGroup(n_a, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_a')
        
        pop_b = NeuronGroup(n_b, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_b')
        
        pop_c = NeuronGroup(n_c, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                            curr_adapt += J_spi
                            ''', refractory='tau_refr', method='euler', name='pop_c')
        
    all_neurons = [pop_t,pop_a,pop_b,pop_c]

    # Rather verbose code assigning values to the parameters of all the populations
    for pop in all_neurons:
        if pop.name == 'pop_t':
            pop.mem_cap = net_params['mem_cap_t'].get_param()
            pop.g_leak = net_params['g_leak_t'].get_param()
            pop.v_reset = net_params['v_reset_t'].get_param()
            pop.tau_adapt = net_params['tau_adapt_t'].get_param()
            pop.curr_bg = net_params['curr_bg_t'].get_param()
            pop.J_spi = net_params['J_spi_t'].get_param()
            pop.v_thres = net_params['v_thres_t'].get_param()
            pop.J_pot = net_params['J_pot_t'].get_param()
            pop.e_rever = net_params['e_rever_t'].get_param()
            pop.g_it = net_params['g_tt'].get_param()
            pop.g_ia = net_params['g_ta'].get_param()
            pop.g_ib = net_params['g_tb'].get_param()
            pop.g_ic = net_params['g_tc'].get_param()

            # Extra parameters for synaptic depression
            if plastic_ee:
                pop.tau_dep = net_params['tau_dep'].get_param()
                pop.eta = net_params['eta'].get_param()

            # The value of heterogeneous parameters deviates from 1 of a customizable std
            if heterogenous:
                std_e_rever=net_params['std_e_rever'].get_param()
                pop.e_rever = net_params['e_rever_t'].get_param() * clip(1 + std_e_rever * randn(), 0, inf)
                std_g_leak=net_params['std_g_leak'].get_param()
                pop.g_leak = net_params['g_leak_t'].get_param() * clip(1 + std_g_leak * randn(), 0, inf)
            
        elif pop.name == 'pop_a':
            pop.mem_cap = net_params['mem_cap_a'].get_param()
            pop.g_leak = net_params['g_leak_a'].get_param()
            pop.v_reset = net_params['v_reset_a'].get_param()
            pop.tau_adapt = net_params['tau_adapt_a'].get_param()
            pop.curr_bg = net_params['curr_bg_a'].get_param()
            pop.v_thres = net_params['v_thres_a'].get_param()
            pop.J_pot = net_params['J_pot_a'].get_param()
            pop.e_rever = net_params['e_rever_a'].get_param()
            pop.J_spi = net_params['J_spi_a'].get_param()
            pop.g_it = net_params['g_at'].get_param()
            pop.g_ia = net_params['g_aa'].get_param()
            pop.g_ib = net_params['g_ab'].get_param()
            pop.g_ic = net_params['g_ac'].get_param()

            if plastic_ee:
                pop.tau_dep = net_params['tau_dep'].get_param()
                pop.eta = net_params['eta'].get_param()

            if heterogenous:
                std_e_rever=net_params['std_e_rever'].get_param()
                pop.e_rever = net_params['e_rever_a'].get_param() * clip(1 + std_e_rever * randn(), 0, inf)
                std_g_leak=net_params['std_g_leak'].get_param()
                pop.g_leak = net_params['g_leak_a'].get_param() * clip(1 + std_g_leak * randn(), 0, inf)
            
        elif pop.name == 'pop_b':
            pop.mem_cap = net_params['mem_cap_b'].get_param()
            pop.g_leak = net_params['g_leak_b'].get_param()
            pop.v_reset = net_params['v_reset_b'].get_param()
            pop.tau_adapt = net_params['tau_adapt_b'].get_param()
            pop.curr_bg = net_params['curr_bg_b'].get_param()
            pop.J_spi = net_params['J_spi_b'].get_param()
            pop.v_thres = net_params['v_thres_b'].get_param()
            pop.J_pot = net_params['J_pot_b'].get_param()
            pop.e_rever = net_params['e_rever_b'].get_param()
            pop.g_it = net_params['g_bt'].get_param()
            pop.g_ia = net_params['g_ba'].get_param()
            pop.g_ib = net_params['g_bb'].get_param()
            pop.g_ic = net_params['g_bc'].get_param()
            
        elif pop.name == 'pop_c':
            pop.mem_cap = net_params['mem_cap_c'].get_param()
            pop.g_leak = net_params['g_leak_c'].get_param()
            pop.v_reset = net_params['v_reset_c'].get_param()
            pop.tau_adapt = net_params['tau_adapt_c'].get_param()
            pop.curr_bg = net_params['curr_bg_c'].get_param()
            pop.J_spi = net_params['J_spi_c'].get_param()
            pop.v_thres = net_params['v_thres_c'].get_param()
            pop.J_pot = net_params['J_pot_c'].get_param()
            pop.e_rever = net_params['e_rever_c'].get_param()
            pop.g_it = net_params['g_ct'].get_param()
            pop.g_ia = net_params['g_ca'].get_param()
            pop.g_ib = net_params['g_cb'].get_param()
            pop.g_ic = net_params['g_cc'].get_param()

        # These parameters are the same for all the populations
        pop.v_stop = net_params['v_stop'].get_param()
        pop.slope = net_params['slope'].get_param()
        pop.tau_refr = net_params['tau_refr'].get_param()
        pop.tau_d_e = net_params['tau_d_e'].get_param()
        pop.tau_d_i = net_params['tau_d_i'].get_param()
        pop.e_e = net_params['e_e'].get_param()
        pop.e_i = net_params['e_i'].get_param()

    # Setting up random connectivity
    prob_tt = net_params['prob_tt'].get_param()
    conn_tt = Connectivity(prob_tt, n_t, n_t, 'conn_tt')
    prob_at = net_params['prob_at'].get_param()
    conn_at = Connectivity(prob_at, n_t, n_a, 'conn_at')
    prob_ta = net_params['prob_ta'].get_param()
    conn_ta = Connectivity(prob_ta, n_a, n_t, 'conn_ta')
    prob_aa = net_params['prob_aa'].get_param()
    conn_aa = Connectivity(prob_aa, n_a, n_a, 'conn_aa')
    prob_tb = net_params['prob_tb'].get_param()
    conn_tb = Connectivity(prob_tb, n_b, n_t, 'conn_tb')
    prob_ab = net_params['prob_ab'].get_param()
    conn_ab = Connectivity(prob_ab, n_b, n_a, 'conn_ab')
    prob_tc = net_params['prob_tc'].get_param()
    conn_tc = Connectivity(prob_tc, n_c, n_t, 'conn_tc')
    prob_ac = net_params['prob_ac'].get_param()
    conn_ac = Connectivity(prob_ac, n_c, n_a, 'conn_ac')
    prob_bt = net_params['prob_bt'].get_param()
    conn_bt = Connectivity(prob_bt, n_t, n_b, 'conn_bt')
    prob_ba = net_params['prob_ba'].get_param()
    conn_ba = Connectivity(prob_ba, n_a, n_b, 'conn_ba')
    prob_ct = net_params['prob_ct'].get_param()
    conn_ct = Connectivity(prob_ct, n_t, n_c, 'conn_ct')
    prob_ca = net_params['prob_ca'].get_param()
    conn_ca = Connectivity(prob_ca, n_a, n_c, 'conn_ca')
    prob_bb = net_params['prob_bb'].get_param()
    conn_bb = Connectivity(prob_bb, n_b, n_b, 'conn_bb')
    prob_cc = net_params['prob_cc'].get_param()
    conn_cc = Connectivity(prob_cc, n_c, n_c, 'conn_cc')
    prob_cb = net_params['prob_cb'].get_param()
    conn_cb = Connectivity(prob_cb, n_b, n_c, 'conn_cb')
    prob_bc = net_params['prob_bc'].get_param()
    conn_bc = Connectivity(prob_bc, n_c, n_b, 'conn_bc')
    
    tau_l = net_params['tau_l'].get_param()

    # Creating synapses with the desired properties
    if plastic_ee:
        # In the depressing network, the effect of EE synapses is scaled by the eff variable of the presynaptic neuron
        syn_tt = Synapses(pop_t, pop_t, on_pre='''g_t += g_it*eff_pre''',
                             delay=tau_l, method='euler',
                            name='syn_tt')
        syn_tt.connect(i=conn_tt.pre_index, j=conn_tt.post_index)
    elif heterogenous:
        # In the heterogeneours network, the effect of each EE synapses is scaled by its own heterogeneous factor
        syn_tt = Synapses(pop_t, pop_t, 'het : 1', on_pre='g_t += g_it*het', name='syn_tt')
        syn_tt.connect(i=conn_tt.pre_index, j=conn_tt.post_index)
        std_cond=net_params['std_cond'].get_param()
        # Values of the heterogeneous factors deviate from 1 of a customizable std
        syn_tt.het = clip(1 + std_cond * randn(), 0, inf)
        std_delay=net_params['std_delay'].get_param()
        # Synaptic delays are also heterogeneous
        syn_tt.delay = tau_l*clip(1 + std_delay * randn(), 0, inf)
    else:
        # In the regular network, each EE synapse increases the value of the postsynaptic conductance from the presynaptic kind (here, t)
        syn_tt = Synapses(pop_t, pop_t, on_pre='g_t += g_it',
                              delay=tau_l, name='syn_tt')
        syn_tt.connect(i=conn_tt.pre_index, j=conn_tt.post_index)

    # In the following: same as before, for the other EE connections
    if plastic_ee:
        syn_at = Synapses(pop_t, pop_a, on_pre='''g_t += (g_it*eff_pre)''',
                             delay=tau_l, method='euler',
                            name='syn_at')
        syn_at.connect(i=conn_at.pre_index, j=conn_at.post_index)
    elif heterogenous:
        syn_at = Synapses(pop_t, pop_a, 'het : 1', on_pre='g_t += g_it*het', name='syn_at')
        syn_at.connect(i=conn_at.pre_index, j=conn_at.post_index)
        std_cond=net_params['std_cond'].get_param()
        syn_at.het = clip(1 + std_cond * randn(), 0, inf)
        std_delay=net_params['std_delay'].get_param()
        syn_at.delay = tau_l*clip(1 + std_delay * randn(), 0, inf)
        
    else:
        syn_at = Synapses(pop_t, pop_a, on_pre='g_t += g_it',
                          delay=tau_l, name='syn_at')
        syn_at.connect(i=conn_at.pre_index, j=conn_at.post_index)
        
    if plastic_ee:
        syn_ta = Synapses(pop_a, pop_t, on_pre='''g_a += (g_ia*eff_pre)''',
                             delay=tau_l, method='euler',
                            name='syn_ta')
        syn_ta.connect(i=conn_ta.pre_index, j=conn_ta.post_index)
    elif heterogenous:
        syn_ta = Synapses(pop_a, pop_t, 'het : 1', on_pre='g_a += g_ia*het', name='syn_pq')
        syn_ta.connect(i=conn_ta.pre_index, j=conn_ta.post_index)
        std_cond=net_params['std_cond'].get_param()
        syn_ta.het = clip(1 + std_cond * randn(), 0, inf)
        std_delay=net_params['std_delay'].get_param()
        syn_ta.delay = tau_l*clip(1 + std_delay * randn(), 0, inf)
    else:
        syn_ta = Synapses(pop_a, pop_t, on_pre='g_a += g_ia',
                          delay=tau_l, name='syn_ta')
        syn_ta.connect(i=conn_ta.pre_index, j=conn_ta.post_index)
        
    if plastic_ee:
        syn_aa = Synapses(pop_a, pop_a, on_pre='''g_a_post += g_ia*eff_pre''',
                             delay=tau_l, method='euler',
                            name='syn_aa')
        syn_aa.connect(i=conn_aa.pre_index, j=conn_aa.post_index)
    elif heterogenous:
        syn_aa = Synapses(pop_a, pop_a, 'het : 1', on_pre='g_a += g_ia*het', name='syn_aa')
        syn_aa.connect(i=conn_aa.pre_index, j=conn_aa.post_index)
        std_cond=net_params['std_cond'].get_param()
        syn_aa.het = clip(1 + std_cond * randn(), 0, inf)
        std_delay=net_params['std_delay'].get_param()
        syn_aa.delay = tau_l*clip(1 + std_delay * randn(), 0, inf)
    else:
        syn_aa = Synapses(pop_a, pop_a, on_pre='g_a += g_ia',
                          delay=tau_l, name='syn_aa')
        syn_aa.connect(i=conn_aa.pre_index, j=conn_aa.post_index)

    # Other synapses are always regular
    syn_tb = Synapses(pop_b, pop_t, on_pre='g_b += g_ib',
                  delay=tau_l, name='syn_tb')
    syn_tb.connect(i=conn_tb.pre_index, j=conn_tb.post_index)
    
    syn_ab = Synapses(pop_b, pop_a, on_pre='g_b += g_ib',
                      delay=tau_l, name='syn_ab')
    syn_ab.connect(i=conn_ab.pre_index, j=conn_ab.post_index)
    
    syn_tc = Synapses(pop_c, pop_t, on_pre='g_c += g_ic',
                      delay=tau_l, name='syn_tc')
    syn_tc.connect(i=conn_tc.pre_index, j=conn_tc.post_index)
    
    syn_ac = Synapses(pop_c, pop_a, on_pre='g_c += g_ic',
                      delay=tau_l, name='syn_ac')
    syn_ac.connect(i=conn_ac.pre_index, j=conn_ac.post_index)
    
    syn_bt = Synapses(pop_t, pop_b, on_pre='g_t += g_it',
                      delay=tau_l, name='syn_bt')
    syn_bt.connect(i=conn_bt.pre_index, j=conn_bt.post_index)
    
    syn_ba = Synapses(pop_a, pop_b, on_pre='g_a += g_ia',
                      delay=tau_l, name='syn_ba')
    syn_ba.connect(i=conn_ba.pre_index, j=conn_ba.post_index)
    
    syn_ct = Synapses(pop_t, pop_c, on_pre='g_t += g_it',
                      delay=tau_l, name='syn_ct')
    syn_ct.connect(i=conn_ct.pre_index, j=conn_ct.post_index)
    
    syn_ca = Synapses(pop_a, pop_c, on_pre='g_a += g_ia',
                      delay=tau_l, name='syn_ca')
    syn_ca.connect(i=conn_ca.pre_index, j=conn_ca.post_index)
    
    syn_bb = Synapses(pop_b, pop_b, on_pre='g_b += g_ib',
                      delay=tau_l, name='syn_bb')
    syn_bb.connect(i=conn_bb.pre_index, j=conn_bb.post_index)
    
    syn_cc = Synapses(pop_c, pop_c, on_pre='g_c += g_ic',
                      delay=tau_l, name='syn_cc')
    syn_cc.connect(i=conn_cc.pre_index, j=conn_cc.post_index)
    
    syn_cb = Synapses(pop_b, pop_c, on_pre='g_b += g_ib',
                      delay=tau_l, name='syn_cb')
    syn_cb.connect(i=conn_cb.pre_index, j=conn_cb.post_index)
    
    syn_bc = Synapses(pop_c, pop_b, on_pre='g_c += g_ic',
                      delay=tau_l, name='syn_bc')
    syn_bc.connect(i=conn_bc.pre_index, j=conn_bc.post_index)

    # Random initialization at values spred in the usual range spanned by voltages and adaptive currents
    pop_t.v = pop_t.e_rever +(20*rand(n_t)-10)*mV
    pop_a.v = pop_a.e_rever +(20*rand(n_a)-10)*mV
    pop_b.v = pop_b.e_rever +(20*rand(n_b)-10)*mV
    pop_c.v = pop_c.e_rever +(20*rand(n_c)-10)*mV
    pop_t.curr_adapt = 150*rand(n_t)*pA
    pop_a.curr_adapt = 150*rand(n_a)*pA
    pop_b.curr_adapt = 150*rand(n_b)*pA
    pop_c.curr_adapt = 50*rand(n_c)*pA

    if plastic_ee:
        pop_t.eff=0.5
        pop_a.eff=0.5
    
    built_network = Network(collect()) # collects all the Brian objects created so far
    used_params = get_used_params(net_params) # collects all the used parameters
    
    return built_network, used_params


def record_network(built_network, used_net_params, test_params):
    '''Sets up "Monitors" to record variables of interest during network simulations'''
    
    test_seed = int(test_params['seed'].get_param())
    rec_adapt_num = int(test_params['rec_adapt_num'].get_param())
    rec_mempo_num = int(test_params['rec_mempo_num'].get_param())
    rec_curr_num = int(test_params['rec_curr_num'].get_param())
    record_spikes = int(test_params['record_spikes'].get_param()) #recording spikes can be switched off if not needed

    # Monitor for the spikes
    pop_t = built_network['pop_t']
    if record_spikes:
        spm_t = SpikeMonitor(pop_t, name='spm_t')
        built_network.add(spm_t)

    # Monitor for the firing rate
    rtm_t = PopulationRateMonitor(pop_t, name='rtm_t')
    built_network.add(rtm_t)

    # Monitors for adaptation, voltages, and b-to-E currents (the latter will be used to estimate the LFP signal)
    stm_t_adp = StateMonitor(pop_t, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_t.N,
                            size=rec_adapt_num, replace=False),name='stm_t_adp')
    built_network.add(stm_t_adp)
    stm_t_mempo = StateMonitor(pop_t, 'v', record=np.random.default_rng(test_seed).choice(pop_t.N,
                            size=rec_mempo_num, replace=False),name='stm_t_mempo')
    built_network.add(stm_t_mempo)
    stm_tb = StateMonitor(pop_t, 'curr_b', record=np.random.default_rng(test_seed).choice(pop_t.N,
                            size=rec_curr_num, replace=False), name='stm_tb')
    built_network.add(stm_tb)

    pop_a = built_network['pop_a']
    if record_spikes:
        spm_a = SpikeMonitor(pop_a, name='spm_a')
        built_network.add(spm_a)
        
    rtm_a = PopulationRateMonitor(pop_a, name='rtm_a')
    built_network.add(rtm_a)
    stm_a_adp = StateMonitor(pop_a, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_a.N,
                            size=rec_adapt_num, replace=False),name='stm_a_adp')
    built_network.add(stm_a_adp)
    stm_a_mempo = StateMonitor(pop_a, 'v', record=np.random.default_rng(test_seed).choice(pop_a.N,
                            size=rec_mempo_num, replace=False),name='stm_a_mempo')
    built_network.add(stm_a_mempo)
    stm_ab = StateMonitor(pop_a, 'curr_b', record=np.random.default_rng(test_seed).choice(pop_a.N,
                            size=rec_curr_num, replace=False), name='stm_ab')
    built_network.add(stm_ab)

        
    pop_b = built_network['pop_b']
    if record_spikes:
        spm_b = SpikeMonitor(pop_b, name='spm_b')
        built_network.add(spm_b)
        
    rtm_b = PopulationRateMonitor(pop_b, name='rtm_b')
    built_network.add(rtm_b)
    stm_b_adp = StateMonitor(pop_b, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_b.N,
                            size=rec_adapt_num, replace=False),name='stm_b_adp')
    built_network.add(stm_b_adp)
    stm_b_mempo = StateMonitor(pop_b, 'v', record=np.random.default_rng(test_seed).choice(pop_b.N,
                            size=rec_mempo_num, replace=False),name='stm_b_mempo')
    built_network.add(stm_b_mempo)
    
    pop_c = built_network['pop_c']
    if record_spikes:
        spm_c = SpikeMonitor(pop_c, name='spm_c')
        built_network.add(spm_c)
        
    rtm_c = PopulationRateMonitor(pop_c, name='rtm_c')
    built_network.add(rtm_c)
    stm_c_adp = StateMonitor(pop_c, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_c.N,
                            size=rec_adapt_num, replace=False),name='stm_c_adp')
    built_network.add(stm_c_adp)
    stm_c_mempo = StateMonitor(pop_c, 'v', record=np.random.default_rng(test_seed).choice(pop_c.N,
                            size=rec_mempo_num, replace=False),name='stm_c_mempo')
    built_network.add(stm_c_mempo)

    return built_network, test_params