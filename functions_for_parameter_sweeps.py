import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from brian2 import *
from network_code.utils import *
from network_code.parameters import *
from network_code.network import *
from network_code.simulations import *
import pickle
import json
import os



def adjust_recurrent_excitation(lfp_size, it, target, exc_cond):
    '''
    Adjusts the EE weights in order to bring the lfp peaks closer to the target size

    Args:
        lfp_size: current size of LFP peaks
        it: current iteration of the balancing algorithm
        target: target size of LFP peaks, chosen as the average LFP peak size in the default parameter set
        exc_cond: current vaule of the EE weights

    Returns:
        exc_cond: new value of the EE weights
    '''

    # values are chosen in order to converge in a reasonable time
    # if the current peak size is far from the target, we get larger adjustments
    # With growing number of iterations, adjustments get smaller, as large amounts of iterations usually occur when we get stuck between a too small and a too large value of exc_cond
    # If the necessary exc_cond value is too far from the default, we recommend to start with larger increases
    if it<20:
        if lfp_size < target-70:
            exc_cond = exc_cond + 0.005

        elif lfp_size > target-70 and lfp_size < target-40:
            exc_cond = exc_cond + 0.002

        elif lfp_size > target-40 and lfp_size < target-20:
            exc_cond = exc_cond + 0.001

        elif lfp_size > target+20 and lfp_size < target+40:
            exc_cond = exc_cond - 0.001

        elif lfp_size > target+40 and lfp_size < target+70:
            exc_cond = exc_cond - 0.002

        elif lfp_size > target+70:
            exc_cond = exc_cond - 0.005
        
    if it>=20 and it<40:
        if lfp_size < target-70:
            exc_cond = exc_cond + 0.0025

        elif lfp_size > target-70 and lfp_size < target-40:
            exc_cond = exc_cond + 0.001

        elif lfp_size > target-40 and lfp_size < target-20:
            exc_cond = exc_cond + 0.0005

        elif lfp_size > target+20 and lfp_size < target+40:
            exc_cond = exc_cond - 0.0005

        elif lfp_size > target+40 and lfp_size < target+70:
            exc_cond = exc_cond - 0.001

        elif lfp_size > target+70:
            exc_cond = exc_cond - 0.0025
            
    if it>=40:
        if lfp_size < target-70:
            exc_cond = exc_cond + 0.001

        elif lfp_size > target-70 and lfp_size < target-40:
            exc_cond = exc_cond + 0.0005

        elif lfp_size > target-40 and lfp_size < target-20:
            exc_cond = exc_cond + 0.00025

        elif lfp_size > target+20 and lfp_size < target+40:
            exc_cond = exc_cond - 0.00025

        elif lfp_size > target+40 and lfp_size < target+70:
            exc_cond = exc_cond - 0.0005

        elif lfp_size > target+70:
            exc_cond = exc_cond - 0.001

    return exc_cond



def analyze_and_prepare_plotting_1parameter(conn1, values=41, plot_frac=0.3, plastic_ee=False, heterogenous=False, plot_second_peak=False):
    '''
    Opens the grouped and analyuzed outputs of 1-parameter sweeps and prepares them for plotting  

    Args:
        conn1: varied connectivity
        values: amount of connectivity values, always 41 for our parameter sweeps
        plot_frac: fraction of the firing rate peaks at which we draw the start and end contour lines for that peak
        plastic_ee: boolean for plastic networks
        heterogenous: boolean for heterogenous networks
        plot_second_peak: boolean for plotting the second peak

    Returns:
        contour_start_a1: array with the starting times of the first athorny peak, as a function of connectivity
        peaktime_a1: array with the peak times of the first athorny peak, as a function of connectivity
        contour_end_a1: array with the end times of the first athorny peak, as a function of connectivity
        contour_start_a2: array with the starting times of the second athorny peak, as a function of connectivity
        peaktime_a2: array with the peak times of the second athorny peak, as a function of connectivity
        contour_end_a2: array with the end times of the second athorny peak, as a function of connectivity
        contour_start_t: array with the starting times of the first thorny peak, as a function of connectivity
        peaktime_t1: array with the peak times of the first thorny peak, as a function of connectivity
        contour_end_t: array with the end times of the first thorny peak, as a function of connectivity
        shifted_a: mean SPW trace for the firing rate of a
        shifted_t: mean SPW trace for the firing rate of t
        zvals11: range to color for the first athorny peak
        zvals12: range to color for the second athorny peak
        zvals2: range to color for the firts thorny peak
        cmap1: colormap for the first athorny peak
        cmap12: colormap for the second athorny peak
        cmap2: colormap for the first thorny peak
    '''

    if plastic_ee:
        with open(f'data/data_dep_{conn1}.pkl','rb') as f:
            loaded_dict = pickle.load(f)
    elif heterogenous:
        with open(f'data/data_het_{conn1}.pkl','rb') as f:
            loaded_dict = pickle.load(f)
    else:
        with open(f'data/data_{conn1}.pkl','rb') as f:
            loaded_dict = pickle.load(f)
    
    peak_argmax_a1 = loaded_dict['analyzed_entries']['peak_argmax_a1']
    peak_argmax_t1 = loaded_dict['analyzed_entries']['peak_argmax_t1']
    peak_argmax_a2 = loaded_dict['analyzed_entries']['peak_argmax_a2']
    peak_argmax_t2 = loaded_dict['analyzed_entries']['peak_argmax_t2']
    peak_a1 = loaded_dict['analyzed_entries']['peak_a1']
    peak_t1 = loaded_dict['analyzed_entries']['peak_t1']
    peak_a2 = loaded_dict['analyzed_entries']['peak_a2']
    peak_t2 = loaded_dict['analyzed_entries']['peak_t2']
    
    rtm_a_trace = loaded_dict['rtm_a_trace']
    rtm_t_trace = loaded_dict['rtm_t_trace']

    # For each connectivity value, time traces for a and t are shifted with respect to the first peak of a, so to always have that peak in the middle
    # To allow for shifting, traces are pasted in a twice-as-long array 
    shifted_a = np.zeros((values,8000))
    for i in range(values):
        shift_value = int(peak_argmax_a1[i])
        shifted_a[i, 4000-shift_value:8000-shift_value] = rtm_a_trace[i,:]
        
    shifted_t = np.zeros((values,8000))
    for i in range(values):
        shift_value = int(peak_argmax_a1[i])
        shifted_t[i, 4000-shift_value:8000-shift_value] = rtm_t_trace[i,:]

    # Peaks are also shifted with respect to the first peak of a, for each connectivity.
    peak_argmax_a2=peak_argmax_a2-peak_argmax_a1
    peak_argmax_t1=peak_argmax_t1-peak_argmax_a1
    peak_argmax_t2=peak_argmax_t2-peak_argmax_a1
    # The a1 peak is shifted last, and takes the value 0 for each connectivity
    peak_argmax_a1=peak_argmax_a1-peak_argmax_a1

    # For each peak, we calculate the corresponding index, so to have 4000 in the middle instead of 0
    peak_idx_a2=(peak_argmax_a2+4000).astype(int)
    peak_idx_t1=(peak_argmax_t1+4000).astype(int)
    peak_idx_t2=(peak_argmax_t2+4000).astype(int)
    peak_idx_a1=(peak_argmax_a1+4000).astype(int)

    # For each peak, we also calculate the corresponding time, with time 0 being the time of the a1 peak
    peaktime_a2=(peak_argmax_a2)/10
    peaktime_t1=(peak_argmax_t1)/10
    peaktime_t2=(peak_argmax_t2)/10
    peaktime_a1=(peak_argmax_a1)/10
    
    contour_start_a1=np.zeros(values)
    contour_start_t=np.zeros(values)
    contour_start_a2=np.zeros(values)
    contour_end_a1=np.zeros(values)
    contour_end_t=np.zeros(values)
    contour_end_a2=np.zeros(values)
    
    fraction = plot_frac
    for i in range(values):

        # Peak a1 starts at the time for which the rate trace is closest to a fraction of the peak, among times before the peak
        contour_start_a1[i] = (np.argmin(np.abs(fraction*peak_a1[i]-shifted_a[i,:peak_idx_a1[i]]))-4000)/10
        # For the start of peak t1, we look before the peak of t1
        contour_start_t[i] = (np.argmin(np.abs(fraction*peak_t1[i]-shifted_t[i,:peak_idx_t1[i]]))-4000)/10
        # For the end of t1, we look after the peak of t1
        contour_end_t[i] = (peak_idx_t1[i]+np.argmin(np.abs(fraction*peak_t1[i]-shifted_t[i,peak_idx_t1[i]:]))-4000)/10
        
        if not np.isnan(peaktime_a2[i]) and (peak_idx_a2[i]>peak_idx_a1[i]):
            # If a has two peaks, we find the lowest value of the a rate between the two peaks
            a_valley_idx = peak_idx_a1[i]+np.argmin(shifted_a[i,peak_idx_a1[i]:peak_idx_a2[i]])
            # Handle extreme case
            if a_valley_idx == peak_idx_a1[i]:
                contour_end_a1[i] = peak_a1[i]
                
            else:
                # For the end of peak a1, we look between the peak of a1 and the "valley"
                contour_end_a1[i] = (peak_idx_a1[i]+np.argmin(np.abs(fraction*peak_a1[i]-shifted_a[i,peak_idx_a1[i]:a_valley_idx]))-4000)/10

            # Handle extreme case
            if a_valley_idx == peak_idx_a2[i]:
                contour_start_a2[i] = peak_a2[i]
                
            else:
                # For the start of peak a2, we look between the "valley" and the peak of a2
                contour_start_a2[i] = (a_valley_idx +np.argmin(np.abs(fraction*peak_a1[i]-shifted_a[i,a_valley_idx:peak_idx_a2[i]]))-4000)/10
            # For the end of peak a2, we look after the peak of a2
            contour_end_a2[i] = (peak_idx_a2[i] +np.argmin(np.abs(fraction*peak_a1[i]-shifted_a[i,peak_idx_a2[i]:]))-4000)/10
        
        else:
            # if there is no a2 peak, for the end of peak a1 we just look after the a1 peak
            contour_end_a1[i] = (peak_idx_a1[i]+np.argmin(np.abs(fraction*peak_a1[i]-shifted_a[i,peak_idx_a1[i]:]))-4000)/10
            contour_start_a2[i] = np.nan
            contour_end_a2[i] = np.nan

    # The following code quite elaborately turns our data into objects that can be plotted in the final Figure 4 (or Figure 4 supplements)
    entries = 2000
    original = values
    long_contour_start_a1 = interpolate_array(contour_start_a1, entries)
    long_contour_end_a1 = interpolate_array(contour_end_a1, entries)
    long_contour_start_t = interpolate_array(contour_start_t, entries)
    long_contour_end_t = interpolate_array(contour_end_t, entries)
    long_contour_start_a2 = interpolate_array(contour_start_a2, entries)
    long_contour_end_a2 = interpolate_array(contour_end_a2, entries)
    replicated_array = np.repeat(peak_a1[0:1], round(entries/(original-1)/2))
    for i in range(1, original - 1):
        replicated_array = np.concatenate((replicated_array, np.repeat(peak_a1[i:i+1], round(entries/(original-1)))))
    long_peak_a1 = np.concatenate((replicated_array, np.repeat(peak_a1[-1:], round(entries/(original-1)/2))))
    replicated_array = np.repeat(peak_a2[0:1], round(entries/(original-1)/2))
    for i in range(1, original - 1):
        replicated_array = np.concatenate((replicated_array, np.repeat(peak_a2[i:i+1], round(entries/(original-1)))))
    long_peak_a2 = np.concatenate((replicated_array, np.repeat(peak_a2[-1:], round(entries/(original-1)/2))))
    replicated_array = np.repeat(peak_t1[0:1], round(entries/(original-1)/2))
    for i in range(1, original - 1):
        replicated_array = np.concatenate((replicated_array, np.repeat(peak_t1[i:i+1], round(entries/(original-1)))))
    long_mean_t_max = np.concatenate((replicated_array, np.repeat(peak_t1[-1:], round(entries/(original-1)/2))))
    
    the_mask_a1=np.zeros((entries,8000))
    the_mask_t=np.zeros((entries,8000))
    the_mask_a2=np.zeros((entries,8000))
    
    for i in range(entries):
        the_mask_a1[i,int(4000+10*long_contour_start_a1[i]):int(4000+10*long_contour_end_a1[i])]=1
        the_mask_t[i,int(4000+10*long_contour_start_t[i]):int(4000+10*long_contour_end_t[i])]=1
        
        if not np.isnan(long_contour_start_a2[i]) and plot_second_peak:
            the_mask_a2[i,int(4000+10*long_contour_start_a2[i]):int(4000+10*long_contour_end_a2[i])]=1
    
    zvals11 =np.tile(long_peak_a1,(8000,1)).T*(the_mask_a1)
    zvals12 =np.tile(long_peak_a2,(8000,1)).T*(the_mask_a2)
    zvals2 =np.tile(long_mean_t_max,(8000,1)).T*the_mask_t
    zvals11[zvals11 < 0] = 0
    zvals12[zvals12 < 0] = 0
    zvals2[zvals2 < 0] = 0
    color1 = colorConverter.to_rgba('white')
    color2 = colorConverter.to_rgba('magenta')
    cmap12 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','goldenrod'],256)
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','goldenrod'],256)
    cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',[color1,color2],256)
    alphas = np.linspace(0, 1, cmap2.N+3)
    cmap1._init()
    cmap1._lut[:,-1] = alphas
    cmap12._init()
    cmap12._lut[:,-1] = alphas
    nan_color = colorConverter.to_rgba('white', 0.0)
    cmap12.set_under(nan_color)
    cmap2._init()
    cmap2._lut[:,-1] = alphas
    
    return contour_start_a1, peaktime_a1, contour_end_a1, contour_start_a2, peaktime_a2, contour_end_a2, contour_start_t, \
            peaktime_t1, contour_end_t, shifted_a, shifted_t, zvals11, zvals12, zvals2, cmap1, cmap12, cmap2



def find_balance_1parameter(conn1, x_idx, target=176, itmax=60, itlen=20, plastic_ee=False, heterogenous=False):
    '''
    Finds a value of the EE synaptic weights, shared between all kinds of EE synapses, that allows to maintain
    LFP events with a peak size close to the default one, to compensate for changes in one connectivity

    Args:
        conn1: varied connectivity
        x_idx: multiprocessing core, here corresponding to one value of conn1
        target: target size of LFP peaks, chosen as the average LFP peak size in the default parameter set
        itmax: maximum amount of iterations allowed
        itlen: simulation time for each iteration (longer gives better estimate of mean LFP size)
        plastic_ee: boolean for plastic networks
        heterogenous: boolean for heterogenous networks
    '''

    # Default parameters
    net_params = get_default_net_params()
    exc_cond = net_params['g_aa'].get_param()  / nS #start around the standard parameter value (0.2)

    # Adjustment of default parameters for synaptic depression
    if plastic_ee:
        net_params['J_spi_a'].set_param(120)
        net_params['J_spi_t'].set_param(180)
        net_params['g_ab'].set_param(2)
        net_params['g_tb'].set_param(0.8)
        net_params['g_bt'].set_param(0.55)
        net_params['g_ba'].set_param(0.7)
        exc_cond = 0.265 #start around the parameter value for depression

    # Adjustment of default parameters for heterogeneity
    elif heterogenous:
        net_params['g_ba'].set_param(0.55)
        net_params['g_bt'].set_param(0.75)
        net_params['g_tb'].set_param(0.7)
        net_params['g_ab'].set_param(2.1)
        net_params['g_cc'].set_param(2)
        net_params['g_cb'].set_param(4.5)
        net_params['curr_bg_t'].set_param(295)

    test_params = get_dft_test_params()
    test_params['sim_time'].set_param(itlen)

    # From 0 to 20%
    x1 = x_idx/200
    
    lfp_size = 0
    it = 0

    # Check the size of LFP peaks and keeps running until it mathces the target, or until the max iteration number is reached
    while (lfp_size < target-20 or lfp_size > target+20) and it<itmax:

        # Updated weights
        net_params['g_aa'].set_param(exc_cond)
        net_params['g_at'].set_param(exc_cond)
        net_params['g_ta'].set_param(exc_cond)
        net_params['g_tt'].set_param(exc_cond)

        # Handle problematic case of 0 connections by equivalently setting their weights to 0
        if x1 > 0.001:
            net_params[f'prob_{conn1}'].set_param(x1)
        else:
            net_params[f'prob_{conn1}'].set_param(0.005)
            net_params[f'g_{conn1}'].set_param(0)

        built_network, all_used_params = build_network(net_params, plastic_ee=plastic_ee, heterogenous=heterogenous)
        built_network, test_params = record_network(built_network, all_used_params, test_params)
        tested_network, test_params = run_simulation(built_network, test_params)
        stats, time_traces, test_params = analyze_simulation(tested_network, test_params)

        if stats['n_events']>0:
            lfp_size = np.mean(stats['spw_amp'])
        it+=1
        
        # Adjust exc_cond to get closer to the LFP target size
        exc_cond = adjust_recurrent_excitation(lfp_size, it, target, exc_cond)

    # Save outcomes
    if plastic_ee:
        savefile_name = f'data/balance_dep_{conn1}{round(1000*x1)}.json'
    elif heterogenous:
        savefile_name = f'data/balance_het_{conn1}{round(1000*x1)}.json'
    else:
        savefile_name = f'data/balance_{conn1}{round(1000*x1)}.json'
    
    my_dict = {'exc_cond':exc_cond, 'it':it, 'lfp_size':lfp_size}
    save_dictionary_to_file(my_dict, savefile_name)
    


def find_balance_2parameters(conn1, conn2, x_idx, target=176, itmax=60, itlen=20):
    '''
    Finds a value of the EE synaptic weights, shared between all kinds of EE synapses, that allows to maintain
    LFP events with a peak size close to the default one, to compensate for changes in two connectivities

    Args:
        conn1: first varied connectivity
        conn2: second varied connectivity
        x_idx: multiprocessing core, here corresponding to one value of conn1 and multiple values of conn2
        target: target size of LFP peaks, chosen as the average LFP peak size in the default parameter set
        itmax: maximum amount of iterations allowed
        itlen: simulation time for each iteration (longer gives better estimate of mean LFP size)
    '''

    # Default parameters
    net_params = get_default_net_params()

    test_params = get_dft_test_params()
    test_params['sim_time'].set_param(itlen)

    for z in range(21):
        # From 0 to 20% for EE connectivities (default 4%, 8%, 11%, 15%)
        if conn1 in ['tt','ta','at','aa']:
            x1 = (x_idx%41)/200
        # From 10 to 30% for other connectivities (default 20%)
        else:
            x1 = (20 + x_idx%41)/200

        # For the second connectivity, values are splitted among two cores
        # Each core runs 21 values, the last of the first overlaps with the first of the second
        if conn2 in ['tt','ta','at','aa']:
            x2 = (z + 20*(x_idx//41))/200
        else:
            x2 = (20 + z + 20*(x_idx//41))/200

        exc_cond = net_params['g_aa'].get_param() / nS #start around the standard parameter value (0.2)
        lfp_size = 0
        it = 0

        # Checks the size of LFP peaks and keeps running until it mathces the target, or until the max iteration number is reached
        while (lfp_size < target-20 or lfp_size > target+20) and it<itmax:

            # Updated weights
            net_params['g_aa'].set_param(exc_cond)
            net_params['g_at'].set_param(exc_cond)
            net_params['g_ta'].set_param(exc_cond)
            net_params['g_tt'].set_param(exc_cond)

            # Handle problematic case of 0 connections by equivalently setting their weights to 0
            if x1 > 0.001:
                net_params[f'prob_{conn1}'].set_param(x1)
            else:
                net_params[f'prob_{conn1}'].set_param(0.005)
                net_params[f'g_{conn1}'].set_param(0)

            if x2 > 0.001:
                net_params[f'prob_{conn2}'].set_param(x2)
            else:
                net_params[f'prob_{conn2}'].set_param(0.005)
                net_params[f'g_{conn2}'].set_param(0)
    
            built_network, all_used_params = build_network(net_params)
            built_network, test_params = record_network(built_network, all_used_params, test_params)
            tested_network, test_params = run_simulation(built_network, test_params)
            stats, time_traces, test_params = analyze_simulation(tested_network, test_params)
    
            if stats['n_events']>0:
                lfp_size = np.mean(stats['spw_amp'])
            it+=1
            # Adjusts exc_cond to get closer to the LFP target size
            exc_cond = adjust_recurrent_excitation(lfp_size, it, target, exc_cond)

        # Save outcomes
        savefile_name = f'data/balance_{conn1}{round(1000*x1)}_{conn2}{round(1000*x2)}.json'
        
        my_dict = {'exc_cond':exc_cond, 'it':it, 'lfp_size':lfp_size}
        save_dictionary_to_file(my_dict, savefile_name)



def interpolate_array(original_array, new_length):
    '''
    Interpolates between given array values to give a higher resolution version of the array

    Args:
        original_array: given array
        new_length: new desired size of the array

    Returns:
        interpolated_values: interpolated array
    '''
    
    original_length = len(original_array)
    # Magnified array
    indices = np.linspace(0, original_length - 1, new_length)
    # Interpolates
    interpolated_values = np.interp(indices, np.arange(original_length), original_array)
    
    return interpolated_values



def open_and_group_data_1parameter(conn1, thr=5, prop_double=0.1, plastic_ee=False, heterogenous=False):
    '''
    Groups and saves simulations outcomes as a function of the connectivity parameter
    Deletes the now unnecessary single files containing the outcomes from each simulation
    Analyze subpeak structures and decides which ones to keep

    Args:
        conn1: brian monitor object
        thr: minimum firing rate of a population at an LFP subpeak in order for that subpeak to count as a subpeak for that population
        prop_double: minimum proportion of two-peaked events necesary for the second peak to be taken into account
        plastic_ee: boolean for plastic networks
        heterogenous: boolean for heterogenous networks
    '''

    maxval = 200
    values = 41
    # Connectivity values as they appear in the file names
    x1_values = np.arange(0, maxval+1, maxval//(values-1))

    loaded_data_dict = {}
    means_dict = {}

    entries_to_keep = ['n_events', 'spw_amp','rtm_a_event_argmax', 'rtm_t_event_argmax','rtm_a_event_max', 'rtm_t_event_max',
                          'area_both', 'area_either', 'ratio', 'area_t_not_a', 'area_a_not_t']
    entries_to_analyze = ['rtm_a_argmax_sub', 'rtm_t_argmax_sub', 'rtm_a_max_sub', 'rtm_t_max_sub', 'spw_idx_sub']
    rtm_a_trace = np.zeros((len(x1_values), 4000))
    rtm_t_trace = np.zeros((len(x1_values), 4000))

    # Loop over connectivity values
    for i, x1 in enumerate(x1_values):
        
        if plastic_ee:
            savefile_name = f'data/results_dep_{conn1}{x1}.pkl'
        elif heterogenous:
            savefile_name = f'data/results_het_{conn1}{x1}.pkl'
        else:
            savefile_name = f'data/results_{conn1}{x1}.pkl'

        with open(savefile_name, 'rb') as file:
            loaded_dict = pickle.load(file)

        loaded_data_dict[x1] = loaded_dict
        # For each i, keep the whole time trace of the a and t rates
        rtm_a_trace[i, :] = loaded_data_dict[x1]['rtm_a_trace']
        rtm_t_trace[i, :] = loaded_data_dict[x1]['rtm_t_trace']
        # Delete the now unnecessary single files
        os.remove(savefile_name)
    
    for entry in entries_to_keep:
        # Average the entries that should be averaged (across SPW events, for each connectivity value
        mean_values = np.array([np.mean(loaded_dict[entry]) for loaded_dict in loaded_data_dict.values()])
        means_dict[entry] = mean_values
    
    for entry in entries_to_analyze:
        # Create workspace variables for the data that requires further processing
        globals()[entry] = [None]*values
        
        for idx, x1 in enumerate(x1_values):
            globals()[entry][idx] = loaded_data_dict[x1][entry]
    
    peak_argmax_a1 = np.zeros((values))
    peak_argmax_a2 = np.zeros((values))
    peak_argmax_t1 = np.zeros((values))
    peak_argmax_t2 = np.zeros((values))
    peak_a1 = np.zeros((values))
    peak_a2 = np.zeros((values))
    peak_t1 = np.zeros((values))
    peak_t2 = np.zeros((values))
    
    peaks_argmax_a1 = [[] for _ in range(values)]
    peaks_argmax_a2 = [[] for _ in range(values)]
    peaks_argmax_t1 = [[] for _ in range(values)]
    peaks_argmax_t2 = [[] for _ in range(values)]
    peaks_a1 = [[] for _ in range(values)]
    peaks_a2 = [[] for _ in range(values)]
    peaks_t1 = [[] for _ in range(values)]
    peaks_t2 = [[] for _ in range(values)]

    # Loop over connectivity values
    for i in range(len(x1_values)):
        
        # Initialize to np.nan
        peaks_argmax_a1[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_argmax_a2[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_argmax_t1[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_argmax_t2[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_a1[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_a2[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_t1[i] = np.zeros([len(spw_idx_sub[i])])*np.nan
        peaks_t2[i] = np.zeros([len(spw_idx_sub[i])])*np.nan

        # Loop over SPWs, for each connectivity value
        for j, event in enumerate(rtm_a_argmax_sub[i]):
            
            # If there is only one subpeak
            if len(event) == 1:
                
                # If for the one subpeak the firing rate of a is big enough
                if rtm_a_max_sub[i][j][0]>thr:
                    
                    # Then, for this SPW, that's the first peak for a, called a1
                    peaks_argmax_a1[i][j] = rtm_a_argmax_sub[i][j][0]
                    peaks_a1[i][j] = rtm_a_max_sub[i][j][0]

                # Else, this SPW is ignored with respect to population a

                # If for the one subpeak the firing rate of t is big enough
                if rtm_t_max_sub[i][j][0]>thr:
                    
                    # Then, for this SPW, that's the first peak for t, called t1
                    peaks_argmax_t1[i][j] = rtm_t_argmax_sub[i][j][0]
                    peaks_t1[i][j] = rtm_t_max_sub[i][j][0]

                # Else, this SPW is ignored with respect to population t

            # If there are multiple subpeaks (meaning two, as cases with three peaks barely occur and we ignore them here)
            elif len(event) > 1:
                
                # If for the first subpeak the firing rate of a is big enough
                if rtm_a_max_sub[i][j][0]>thr:
                    
                    # Then, for this SPW, that's the first peak for a, called a1
                    peaks_argmax_a1[i][j] = rtm_a_argmax_sub[i][j][0]
                    peaks_a1[i][j] = rtm_a_max_sub[i][j][0]

                    # And the second one is the second peak for a, called a2
                    peaks_argmax_a2[i][j] = rtm_a_argmax_sub[i][j][1]
                    peaks_a2[i][j] = rtm_a_max_sub[i][j][1]
    
                else:
                    # Otherwise, the second LFP peak is the first peak for a, called a1
                    peaks_argmax_a1[i][j] = rtm_a_argmax_sub[i][j][1]
                    peaks_a1[i][j] = rtm_a_max_sub[i][j][1]

                # If for the first subpeak the firing rate of t is big enough
                if rtm_t_max_sub[i][j][0]>thr:

                    # Then, for this SPW, that's the first peak for t, called t1
                    peaks_argmax_t1[i][j] = rtm_t_argmax_sub[i][j][0]
                    peaks_t1[i][j] = rtm_t_max_sub[i][j][0]
                    
                    # And the second one is the second peak for t, called t2
                    peaks_argmax_t2[i][j] = rtm_t_argmax_sub[i][j][1]
                    peaks_t2[i][j] = rtm_t_max_sub[i][j][1]
    
                else:
                    # Otherwise, the second LFP peak is the first peak for t, called t1
                    peaks_argmax_t1[i][j] = rtm_t_argmax_sub[i][j][1]
                    peaks_t1[i][j] = rtm_t_max_sub[i][j][1]

        # Mean over SPWs, excluding np.nans (so excluding also values for which the firist of that population wasn't sufficient
        peak_argmax_a1[i] = np.nanmean(peaks_argmax_a1[i])
        peak_a1[i] = np.nanmean(peaks_a1[i])
        peak_argmax_t1[i] = np.nanmean(peaks_argmax_t1[i])
        peak_t1[i] = np.nanmean(peaks_t1[i])

        # Second peaks are only averaged to become their own entry if the proportion of SPWs with those peaks is at least prop_double 
        if np.count_nonzero(~np.isnan(peaks_argmax_a2[i])) > prop_double*len(peaks_argmax_a2[i]):
            peak_argmax_a2[i] = np.nanmean(peaks_argmax_a2[i])
            peak_a2[i] = np.nanmean(peaks_a2[i])
        else:
            peak_argmax_a2[i] = np.nan
            peak_a2[i] = np.nan
    
        if np.count_nonzero(~np.isnan(peaks_argmax_t2[i])) > prop_double*len(peaks_argmax_t2[i]):
            peak_argmax_t2[i] = np.nanmean(peaks_argmax_t2[i])
            peak_t2[i] = np.nanmean(peaks_t2[i])
        else:
            peak_argmax_t2[i] = np.nan
            peak_t2[i] = np.nan
    
    analyzed_entries = {'peak_argmax_a1': peak_argmax_a1, 'peak_argmax_t1': peak_argmax_t1,'peak_argmax_a2': peak_argmax_a2,
            'peak_argmax_t2': peak_argmax_t2,'peak_a1': peak_a1, 'peak_t1': peak_t1,
            'peak_a2': peak_a2,'peak_t2': peak_t2}
    all_the_data = {'means_dict': means_dict, 'analyzed_entries': analyzed_entries, 'rtm_a_trace': rtm_a_trace, 'rtm_t_trace': rtm_t_trace}

    # Save everything in one place
    if plastic_ee:
        with open(f'data/data_dep_{conn1}.pkl','wb') as f:
            pickle.dump(all_the_data,f)
    elif heterogenous:
        with open(f'data/data_het_{conn1}.pkl','wb') as f:
            pickle.dump(all_the_data,f)
    else:
        with open(f'data/data_{conn1}.pkl','wb') as f:
            pickle.dump(all_the_data,f)



def open_and_group_data_2parameters(conn1, conn2, thr=5, prop_double=0.1):
    '''
    Groups and saves simulations outcomes as a function of the connectivity parameters
    Deletes the now unnecessary single files containing the outcomes from each simulation
    Analyze subpeak structures and decides which ones to keep

    Args:
        conn1: first varied connectivity
        conn2: second varied connectivity
        thr: minimum firing rate of a population at an LFP subpeak in order for that subpeak to count as a subpeak for that population
        prop_double: minimum proportion of two-peaked events necesary for the second peak to be taken into account
    '''

    values = 41
    # Connectivity values as they appear in the file names
    if conn1 in ['tt','ta','at','aa']:
        maxval = 200
        x1_values = range(0, maxval+1, maxval//(values-1))
    else:
        maxval = 300
        x1_values = range(100, maxval+1, (maxval-100)//(values-1))
    
    if conn2 in ['tt','ta','at','aa']:
        maxval = 200
        x2_values = range(0, maxval+1, maxval//(values-1))
    else:
        maxval = 300
        x2_values = range(100, maxval+1, (maxval-100)//(values-1))
    
    loaded_data_dict = {}
    means_dict = {}
    
    entries_to_keep = ['area_both', 'area_either', 'area_t_not_a', 'area_a_not_t']
    entries_to_analyze = ['rtm_a_argmax_sub', 'rtm_t_argmax_sub', 'rtm_a_max_sub', 'rtm_t_max_sub', 'spw_idx_sub']

    # Loop over connectivity values
    for i, x1 in enumerate(x1_values):
        for j, x2 in enumerate(x2_values):
            savefile_name = f'data/results_{conn1}{x1}_{conn2}{x2}.pkl'
    
            with open(savefile_name, 'rb') as file:
                loaded_dict = pickle.load(file)
    
            loaded_data_dict[(x1, x2)] = loaded_dict
            # Delete the now unnecessary single files
            os.remove(savefile_name)
    
    
    for entry in entries_to_keep:
        the_mean = np.zeros((values, values))
        for i, x1 in enumerate(x1_values):
            for j, x2 in enumerate(x2_values):
                if entry in loaded_data_dict[(x1, x2)]:
                    # Average the entries that should be averaged (across SPW events, for each connectivity value
                    the_mean[i, j] = np.mean(loaded_data_dict[(x1, x2)][entry])
                else:
                    the_mean[i, j] = np.nan
        means_dict[entry] = the_mean
    
    for entry in entries_to_analyze:
        globals()[entry] = [[None]*values for _ in range(values)]
        
        for idx, x1 in enumerate(x1_values):
            for idx2, x2 in enumerate(x2_values):
                if entry in loaded_data_dict[(x1, x2)]:
                # Create workspace variables for the data that requires further processing:
                    globals()[entry][idx][idx2] = loaded_data_dict[(x1, x2)][entry]
                else:
                    globals()[entry][idx][idx2] = []
    
    peak_argmax_a1 = np.zeros((values,values))
    peak_argmax_a2 = np.zeros((values,values))
    peak_argmax_t1 = np.zeros((values,values))
    peak_argmax_t2 = np.zeros((values,values))
    peak_a1 = np.zeros((values,values))
    peak_a2 = np.zeros((values,values))
    peak_t1 = np.zeros((values,values))
    peak_t2 = np.zeros((values,values))
    
    peaks_argmax_a1 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_argmax_a2 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_argmax_t1 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_argmax_t2 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_a1 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_a2 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_t1 = [[[] for _ in range(values)] for _ in range(values)]
    peaks_t2 = [[[] for _ in range(values)] for _ in range(values)]

    # Loop over connectivity values
    for i in range(len(x1_values)):
        for k in range(len(x2_values)):

            # Initialize to np.nan
            peaks_argmax_a1[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_argmax_a2[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_argmax_t1[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_argmax_t2[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_a1[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_a2[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_t1[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan
            peaks_t2[i][k] = np.zeros([len(spw_idx_sub[i][k])])*np.nan

            # Loop over SPWs, for each connectivity values
            for j, event in enumerate(rtm_a_argmax_sub[i][k]):

                # If there is only one subpeak
                if len(event) == 1:

                    # If for the one subpeak the firing rate of a is big enough
                    if rtm_a_max_sub[i][k][j][0]>thr:

                        # Then, for this SPW, that's the first peak for a, called a1
                        peaks_argmax_a1[i][k][j] = rtm_a_argmax_sub[i][k][j][0]
                        peaks_a1[i][k][j] = rtm_a_max_sub[i][k][j][0]

                    # Else, this SPW is ignored with respect to population a

                    # If for the one subpeak the firing rate of t is big enough
                    if rtm_t_max_sub[i][k][j][0]>thr:

                        # Then, for this SPW, that's the first peak for t, called t1
                        peaks_argmax_t1[i][k][j] = rtm_t_argmax_sub[i][k][j][0]
                        peaks_t1[i][k][j] = rtm_t_max_sub[i][k][j][0]

                    # Else, this SPW is ignored with respect to population t

                # If there are multiple subpeaks (meaning two, as cases with three peaks barely occur and we ignore them here)
                elif len(event) > 1:

                    # If for the first subpeak the firing rate of a is big enough
                    if rtm_a_max_sub[i][k][j][0]>thr:

                        # Then, for this SPW, that's the first peak for a, called a1
                        peaks_argmax_a1[i][k][j] = rtm_a_argmax_sub[i][k][j][0]
                        peaks_a1[i][k][j] = rtm_a_max_sub[i][k][j][0]

                        # And the second one is the second peak for a, called a2
                        peaks_argmax_a2[i][k][j] = rtm_a_argmax_sub[i][k][j][1]
                        peaks_a2[i][k][j] = rtm_a_max_sub[i][k][j][1]
        
                    else:
                        # Otherwise, the second LFP peak is the first peak for a, called a1
                        peaks_argmax_a1[i][k][j] = rtm_a_argmax_sub[i][k][j][1]
                        peaks_a1[i][k][j] = rtm_a_max_sub[i][k][j][1]

                    # If for the first subpeak the firing rate of t is big enough
                    if rtm_t_max_sub[i][k][j][0]>thr:

                        # Then, for this SPW, that's the first peak for t, called t1
                        peaks_argmax_t1[i][k][j] = rtm_t_argmax_sub[i][k][j][0]
                        peaks_t1[i][k][j] = rtm_t_max_sub[i][k][j][0]

                        # And the second one is the second peak for t, called t2
                        peaks_argmax_t2[i][k][j] = rtm_t_argmax_sub[i][k][j][1]
                        peaks_t2[i][k][j] = rtm_t_max_sub[i][k][j][1]
        
                    else:
                        # Otherwise, the second LFP peak is the first peak for t, called t1
                        peaks_argmax_t1[i][k][j] = rtm_t_argmax_sub[i][k][j][1]
                        peaks_t1[i][k][j] = rtm_t_max_sub[i][k][j][1]

            # Mean over SPWs, excluding np.nans (so excluding also values for which the firist of that population wasn't sufficient
            peak_argmax_a1[i][k] = np.nanmean(peaks_argmax_a1[i][k])
            peak_a1[i][k] = np.nanmean(peaks_a1[i][k])
            peak_argmax_t1[i][k] = np.nanmean(peaks_argmax_t1[i][k])
            peak_t1[i][k] = np.nanmean(peaks_t1[i][k])

            # Second peaks are only averaged to become their own entry if the proportion of SPWs with those peaks is at least prop_double 
            if np.count_nonzero(~np.isnan(peaks_argmax_a2[i][k])) > prop_double*len(peaks_argmax_a2[i][k]):
                peak_argmax_a2[i][k] = np.nanmean(peaks_argmax_a2[i][k])
                peak_a2[i][k] = np.nanmean(peaks_a2[i][k])
            else:
                peak_argmax_a2[i][k] = np.nan
                peak_a2[i][k] = np.nan
        
            if np.count_nonzero(~np.isnan(peaks_argmax_t2[i][k])) > prop_double*len(peaks_argmax_t2[i][k]):
                peak_argmax_t2[i][k] = np.nanmean(peaks_argmax_t2[i][k])
                peak_t2[i][k] = np.nanmean(peaks_t2[i][k])
            else:
                peak_argmax_t2[i][k] = np.nan
                peak_t2[i][k] = np.nan
    
    analyzed_entries = {'peak_argmax_a1': peak_argmax_a1, 'peak_argmax_t1': peak_argmax_t1}
    all_the_data = {'means_dict': means_dict, 'analyzed_entries': analyzed_entries}

    # Save everything in one place
    with open(f'data/data_{conn1}_{conn2}.pkl','wb') as f:
        pickle.dump(all_the_data,f)



def prepare_plotting_2parameters(conn1, conn2, values=41, lim=0.05):
    '''
    Calculates the delay between the first peak of a and the first peak of t

    Args:
        conn1: first varied connectivity
        conn2: second varied connectivity
        values: amount of connectivity values, always 41 for our parameter sweeps
        lim: minimum amount of participation in the SPW required to both populations

    Returns:
        peak_separation: delay between the peaks, with plus sign indicating athorny before thorny
    '''

    with open(f'data/data_{conn1}_{conn2}.pkl','rb') as f:
        loaded_dict = pickle.load(f)

    peak_argmax_a1 = loaded_dict['analyzed_entries']['peak_argmax_a1']/10-200 #Rescaled in units of time
    peak_argmax_t1 = loaded_dict['analyzed_entries']['peak_argmax_t1']/10-200 #Rescaled in units of time
    area_both = loaded_dict['means_dict']['area_both']
    area_a_not_t = loaded_dict['means_dict']['area_a_not_t']
    area_t_not_a = loaded_dict['means_dict']['area_t_not_a']
    area_either = loaded_dict['means_dict']['area_either']
    # Discards connectivity values for which the participation of either population a or t is less than 5% of the total firing
    # These values are discarded because peaks aren't very informative if their size is so small, which happens for some configurations
    peak_argmax_t1[(area_both+area_a_not_t)/area_either<lim]=np.nan
    peak_argmax_t1[(area_both+area_t_not_a)/area_either<lim]=np.nan
    peak_separation = peak_argmax_t1-peak_argmax_a1
    
    return peak_separation



def save_conductances_1parameter(conn1, plastic_ee=False, heterogenous=False):
    '''
    Groups and saves find_balance outcomes as a function of the connectivity parameter
    Deletes the now unnecessary single files containing the outcomes from each simulation

    Args:
        conn1: varied connectivity
        plastic_ee: boolean for plastic networks
        heterogenous: boolean for heterogenous networks
    '''

    maxval = 200
    values = 41
    exc_cond_array = np.full(values, np.nan)
    it_array = np.full(values, np.nan)
    lfp_size_array = np.full(values, np.nan)

    # Open the saved balance data
    for i in range(0, maxval+1, maxval//(values-1)):
        if plastic_ee:
            file_name = f'data/balance_dep_{conn1}{i}.json'
        elif heterogenous:
            file_name = f'data/balance_het_{conn1}{i}.json'
        else:
            file_name = f'data/balance_{conn1}{i}.json'
        
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
    
            index = i // 5
            exc_cond_array[index] = data.get("exc_cond", np.nan)
            it_array[index] = data.get("it", np.nan)
            lfp_size_array[index] = data.get("lfp_size", np.nan)
            # Remove the single balance files
            os.remove(file_name)
        
        except FileNotFoundError:
            print(f"File {file_name} not found.")
       
    # Save everything in one place
    if plastic_ee:
        file_path = f'data/exc_cond_dep_{conn1}.npy'
    elif heterogenous:
        file_path = f'data/exc_cond_het_{conn1}.npy'
    else:
        file_path = f'data/exc_cond_{conn1}.npy'
    np.save(file_path, exc_cond_array)



def save_conductances_2parameters(conn1, conn2):
    '''
    Groups and saves find_balance outcomes as a function of the connectivity parameter
    Deletes the now unnecessary single files containing the outcomes from each simulation

    Args:
        conn1: first varied connectivity
        conn2: second varied connectivity
    '''

    values = 41
    exc_cond_array = np.full((values, values), np.nan)
    it_array = np.full((values, values), np.nan)
    lfp_size_array = np.full((values, values), np.nan)

    
    if conn1 in ['tt','ta','at','aa']:
        maxval = 200
        range_i = range(0, maxval+1, maxval//(values-1))
    else:
        maxval = 300
        range_i = range(100, maxval+1, (maxval-100)//(values-1))

    if conn2 in ['tt','ta','at','aa']:
        maxval = 200
        range_j = range(0, maxval+1, maxval//(values-1))
    else:
        maxval = 300
        range_j = range(100, maxval+1, (maxval-100)//(values-1))

    # Open the saved balance data
    for i in range_i:
        for j in range_j:
            file_name = f'data/balance_{conn1}{i}_{conn2}{j}.json'
            
            try:
                with open(file_name, 'r') as file:
                    data = json.load(file)
        
                index_i = i // 5 -20
                index_j = j // 5
                exc_cond_array[index_i,index_j] = data.get("exc_cond", np.nan)
                it_array[index_i,index_j] = data.get("it", np.nan)
                lfp_size_array[index_i,index_j] = data.get("lfp_size", np.nan)
                # Remove the single balance files
                os.remove(file_name)
                
            except FileNotFoundError:
                print(f"File {file_name} not found.")
       
    # Save everything in one place
    file_path = f'data/exc_cond_{conn1}_{conn2}.npy'
    np.save(file_path, exc_cond_array)



def save_dictionary_to_file(dictionary, filename):
    '''A handy function to save stuff in json format'''
    
    with open(filename, 'w') as file:
        json.dump(dictionary, file)



def simulate_1parameter(conn1, x_idx, simlen=200, plastic_ee=False, heterogenous=False):
    '''
    Runs a simulation for a given connectivity value and the corresponding EE synaptic weight scaling
    Saves the outcome dictionary "stats"

    Args:
        conn1: varied connectivity
        x_idx: multiprocessing core, here corresponding to one value of conn1
        simlen: duration of the simulation
        plastic_ee: boolean for plastic networks
        heterogenous: boolean for heterogenous networks
    '''

    # Default parameters
    net_params = get_default_net_params()

    # Adjustment of default parameters for synaptic depression
    if plastic_ee:
        net_params['J_spi_a'].set_param(120)
        net_params['J_spi_t'].set_param(180)
        net_params['g_ab'].set_param(2)
        net_params['g_tb'].set_param(0.8)
        net_params['g_bt'].set_param(0.55)
        net_params['g_ba'].set_param(0.7)

    # Adjustment of default parameters for heterogeneity
    elif heterogenous:
        net_params['g_ba'].set_param(0.55)
        net_params['g_bt'].set_param(0.75)
        net_params['g_tb'].set_param(0.7)
        net_params['g_ab'].set_param(2.1)
        net_params['g_cc'].set_param(2)
        net_params['g_cb'].set_param(4.5)
        net_params['curr_bg_t'].set_param(295)

    test_params = get_dft_test_params()
    test_params['sim_time'].set_param(simlen)

    # From 0 to 20%
    x1 = (x_idx%41)/200

    # Handle problematic case of 0 connections by equivalently setting their weights to 0
    if x1 > 0.001:
        net_params[f'prob_{conn1}'].set_param(x1)
    else:
        net_params[f'prob_{conn1}'].set_param(0.005)
        net_params[f'g_{conn1}'].set_param(0)

    # Open the corresponding exc_cond value to balance the network
    if plastic_ee:
        exc_cond = np.load(f'data/exc_cond_dep_{conn1}.npy')[x_idx%41]
    elif heterogenous:
        exc_cond = np.load(f'data/exc_cond_het_{conn1}.npy')[x_idx%41]
    else:
        exc_cond = np.load(f'data/exc_cond_{conn1}.npy')[x_idx%41]

    # Assign exc_cond to all the EE weights
    net_params['g_aa'].set_param(exc_cond)
    net_params['g_at'].set_param(exc_cond)
    net_params['g_ta'].set_param(exc_cond)
    net_params['g_tt'].set_param(exc_cond)
    
    built_network, all_used_params = build_network(net_params, plastic_ee=plastic_ee, heterogenous=heterogenous)
    built_network, test_params = record_network(built_network, all_used_params, test_params)
    tested_network, test_params = run_simulation(built_network, test_params)
    stats, time_traces, test_params = analyze_simulation(tested_network, test_params)

    # Save the stats dictionary from the simulation outcomes
    if plastic_ee:
        savefile_name = f'data/results_dep_{conn1}{round(1000*x1)}.pkl'
    elif heterogenous:
        savefile_name = f'data/results_het_{conn1}{round(1000*x1)}.pkl'
    else:
        savefile_name = f'data/results_{conn1}{round(1000*x1)}.pkl'
    
    with open(savefile_name, 'wb') as file:
        pickle.dump(stats, file)
    


def simulate_2parameters(conn1, conn2, x_idx, simlen=200):
    '''
    Runs a simulation for a certain subset of connectivity combinations and the corresponding EE synaptic weight scaling
    Saves the outcome dictionary "stats"

    Args:
        conn1: first varied connectivity
        conn2: second varied connectivity
        x_idx: multiprocessing core, here corresponding to one value of conn1 and multiple values of conn2
        simlen: duration of the simulation
    '''

    # Default parameters
    net_params = get_default_net_params()

    test_params = get_dft_test_params()
    test_params['sim_time'].set_param(simlen)

    for z in range(21):

        # From 0 to 20% for EE connectivities (default 4%, 8%, 11%, 15%)
        if conn1 in ['tt','ta','at','aa']:
            x1 = (x_idx%41)/200
        # From 10 to 30% for other connectivities (default 20%)
        else:
            x1 = (20 + x_idx%41)/200

        # For the second connectivity, values are splitted among two cores
        # Each core runs 21 values, the last of the first overlaps with the first of the second
        if conn2 in ['tt','ta','at','aa']:
            x2 = (z + 20*(x_idx//41))/200
        else:
            x2 = (20 + z + 20*(x_idx//41))/200

        # Handle problematic case of 0 connections by equivalently setting their weights to 0
        if x1 > 0.001:
            net_params[f'prob_{conn1}'].set_param(x1)
        else:
            net_params[f'prob_{conn1}'].set_param(0.005)
            net_params[f'g_{conn1}'].set_param(0)

        if x2 > 0.001:
            net_params[f'prob_{conn2}'].set_param(x2)
        else:
            net_params[f'prob_{conn2}'].set_param(0.005)
            net_params[f'g_{conn2}'].set_param(0)

        # Open the corresponding exc_cond value to balance the network
        exc_cond = np.load(f'data/exc_cond_{conn1}_{conn2}.npy')[x_idx%41,z]
        
        # Assign exc_cond to all the EE weights
        net_params['g_aa'].set_param(exc_cond)
        net_params['g_at'].set_param(exc_cond)
        net_params['g_ta'].set_param(exc_cond)
        net_params['g_tt'].set_param(exc_cond)
        
        built_network, all_used_params = build_network(net_params)
        built_network, test_params = record_network(built_network, all_used_params, test_params)
        tested_network, test_params = run_simulation(built_network, test_params)
        stats, time_traces, test_params = analyze_simulation(tested_network, test_params)

        # Save the stats dictionary from the simulation outcomes
        savefile_name = f'data/results_{conn1}{round(1000*x1)}_{conn2}{round(1000*x2)}.pkl'
        
        with open(savefile_name, 'wb') as file:
            pickle.dump(stats, file)
