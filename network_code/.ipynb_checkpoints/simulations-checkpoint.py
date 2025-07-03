from brian2 import *
from my_code.parameters import *
#from imported_code.detect_peaks import detect_peaks
from my_code.utils import detect_peaks, check_network_state, check_brian_monitor,\
    calc_network_frequency, trim_brian_monitor, calc_low_pass_filter, calc_unit_rates, \
    calc_isi_cv, get_q_factor, get_newest_file

def run_test(network, test_params):

    prep_time = test_params['prep_time'].get_param()
    sim_time = test_params['sim_time'].get_param()
    sim_dt = test_params['sim_dt'].get_param()
    defaultclock.dt = sim_dt
    network.run(prep_time)
    network.run(sim_time, report='text')

    return network, test_params
    

def prepare_test(network, test_params):

    sim_dt = test_params['sim_dt'].get_param()
    prep_time = test_params['prep_time'].get_param()
    record_spikes = int(test_params['record_spikes'].get_param())
    check_second_peaks = test_params['check_second_peaks'].get_param()
    spw_window = test_params['spw_window'].get_param()
    baseline_start = test_params['baseline_start'].get_param()
    baseline_end = test_params['baseline_end'].get_param()
    min_peak_dist = test_params['min_peak_dist'].get_param()
    lfp_cutoff = test_params['lfp_cutoff'].get_param()
    min_peak_height = test_params['min_peak_height'].get_param() / pA
    min_subpeak_dist = test_params['min_subpeak_dist'].get_param()
    gauss_window = test_params['gauss_window'].get_param()
    
    start_time = 0*second
    start_stats = prep_time
    the_time = network['rtm_t'].t / second
    end_time = np.max(the_time) * second
    time_traces = {'the_time':the_time}
    stats = {'start_time':start_time, 'start_stats':start_stats, 'end_time':end_time}

    if record_spikes:
        spm_array = ['spm_t', 'spm_a', 'spm_b', 'spm_c']
    
        for spm in spm_array:
            time_traces[spm] = trim_brian_monitor(network[spm], network[spm].i, 1, start_time, end_time)
    
    # our LFP is the negative mean B->P current:
    t_num = network['pop_t'].N
    a_num = network['pop_a'].N
    b_to_exc_current = - a_num * np.mean(network['stm_ab'].curr_b, axis=0) / (a_num + t_num) - t_num * np.mean(network['stm_tb'].curr_b, axis=0) / (a_num + t_num)
    time_traces['lfp'] = b_to_exc_current
    lfp_time, lfp_trace = trim_brian_monitor(network['stm_ab'], b_to_exc_current, pA, start_time, end_time)
    lowpass_lfp = calc_low_pass_filter(lfp_trace, lfp_cutoff / Hz, sim_dt / second)
    time_traces['lowpass_lfp'] = lowpass_lfp
        
    mpd = int(min_peak_dist / sim_dt)
    peak_idx = detect_peaks(lowpass_lfp, mph=min_peak_height, mpd=int(min_peak_dist / sim_dt))
    peak_idx = peak_idx[lfp_time[peak_idx] >= (start_stats / second)]
    peak_idx = peak_idx[:-1]
    n_events = len(peak_idx)
    stats['n_events'] = n_events

    if n_events >= 1:

        # calculate LFP baseline as the mean of the [-200,-100] ms preceding peaks:
        baseline_start = int(-baseline_start / sim_dt)
        baseline_end = int(-baseline_end / sim_dt)
        baseline_window = np.array([range(peak + baseline_start, peak + baseline_end + 1) for peak in peak_idx], dtype=int)
        baseline = np.mean(lowpass_lfp[baseline_window])
        
        # get amplitude of spws:
        spw_amp = lowpass_lfp[peak_idx]
        stats['spw_amp'] = spw_amp

        # calculate times at half-maximum to get event start and end times:
        spw_halfmax = baseline + (spw_amp - baseline) / 2.0
        event_pre_index = np.zeros(n_events, dtype=int)
        event_post_index = np.zeros(n_events, dtype=int)
        window = int(spw_window / sim_dt)
        
        for i in range(n_events):
            # find event start:
            aux_idx = (np.abs(lowpass_lfp[peak_idx[i] - window : peak_idx[i]] - spw_halfmax[i])).argmin()
            event_pre_index[i] = int(aux_idx + peak_idx[i] - window)

            # find event end:
            aux_idx = (np.abs(lowpass_lfp[peak_idx[i]: peak_idx[i] + window] - spw_halfmax[i])).argmin()
            event_post_index[i] = int(aux_idx + peak_idx[i])

        # get FWHM (duration) of spws:
        spw_fwhm = lfp_time[event_post_index] - lfp_time[event_pre_index]
        stats['event_durations'] = spw_fwhm * 1e3  # in ms

        # get Inter-Event-Interval of spws:
        spw_iei = lfp_time[event_pre_index[1:]] - lfp_time[event_post_index[:-1]]
        stats['event_intervals'] = spw_iei

        stats['event_pre_index'] = event_pre_index
        stats['event_peak_index'] = peak_idx
        stats['event_post_index'] = event_post_index

        # subpeaks
        subpeak_amp = [np.array([]) for _ in range(n_events)]
        subpeak_idx = [np.array([]) for _ in range(n_events)]
        
        if check_second_peaks:
            for i in range(n_events):
                event_trace = lowpass_lfp[peak_idx[i] - window:peak_idx[i] + window]
                subpeaks = detect_peaks(event_trace, mph=min_peak_height, mpd=int(min_subpeak_dist / sim_dt))
                subpeak_idx[i] = peak_idx[i] - window + subpeaks
                subpeak_amp[i] = lowpass_lfp[subpeak_idx[i]]
                
        stats['spw_amp_sub'] = subpeak_amp
        stats['spw_idx_sub'] = subpeak_idx

    rtm_array = ['rtm_t', 'rtm_a', 'rtm_b', 'rtm_c']
    adapt_array = ['stm_t_adp', 'stm_a_adp', 'stm_b_adp', 'stm_c_adp']
    mempo_array = ['stm_t_mempo', 'stm_a_mempo', 'stm_b_mempo', 'stm_c_mempo']
    
    for rtm in rtm_array: 
        time_traces[rtm] = trim_brian_monitor(network[rtm],network[rtm].smooth_rate('gaussian', width=gauss_window), Hz, start_time, end_time)[1]
        the_rate = time_traces[rtm]
        
        if n_events > 0:
            baseline_rate = np.mean(the_rate[baseline_window])
            event_mean_firing = np.zeros(n_events)
            nspw_mean_firing = np.zeros(n_events-1)
            event_argmax_firing = np.zeros(n_events)
            event_max_firing = np.zeros(n_events)
            
            for i in range(n_events):
                win_start = int(peak_idx[i]- window)
                win_end = int(peak_idx[i]+ window)
                event_mean_firing[i] = np.mean(the_rate[win_start:win_end])
                event_argmax_firing[i] = np.argmax(the_rate[win_start:win_end])
                event_max_firing[i] = np.max(the_rate[win_start:win_end])

            for i in range(n_events-1):
                nspw_mean_firing[i] = np.mean(the_rate[event_post_index[i]:event_pre_index[i+1]])
                
            firing_begin = the_rate[event_pre_index]
            firing_peak = the_rate[peak_idx]
            firing_end = the_rate[event_post_index]

            stats[rtm + '_event'] = event_mean_firing
            stats[rtm + '_nspw'] = nspw_mean_firing
            stats[rtm + '_event_argmax'] = event_argmax_firing
            stats[rtm + '_event_max'] = event_max_firing
            stats[rtm + '_begin'] = firing_begin
            stats[rtm + '_peak'] = firing_peak
            stats[rtm + '_end'] = firing_end
            
            # subpeaks
            if rtm == 'rtm_t' or rtm == 'rtm_a':
                window = int(spw_window / sim_dt) 
                spw_traces = np.zeros((n_events, 2*window))
                
                for i in range(n_events):
                    win_start = int(peak_idx[i]- window)
                    win_end = int(peak_idx[i]+ window)
                    spw_traces[i,:] = the_rate[win_start:win_end]
                
                spw_trace = np.mean(spw_traces,0)
                time_traces[rtm + '_trace'] = spw_trace
                
                if check_second_peaks:
                    sub_window = int(min_subpeak_dist / sim_dt)
                    sub_event_argmax_firing = [np.array([]) for _ in range(n_events)]
                    sub_event_max_firing = [np.array([]) for _ in range(n_events)]
                    
                    for i in range(n_events):
                        sub_event_argmax_firing[i] = np.zeros(len(subpeak_idx[i]))
                        sub_event_max_firing[i] = np.zeros(len(subpeak_idx[i]))
                        
                        for j in range(len(subpeak_idx[i])):
                            win_start = int(subpeak_idx[i][j]- sub_window)
                            win_end = int(subpeak_idx[i][j]+ sub_window)
                            sub_event_argmax_firing_subtime = np.argmax(the_rate[win_start:win_end])
                            sub_event_max_firing[i][j] = np.max(the_rate[win_start:win_end])
                            sub_event_argmax_firing[i][j] = sub_event_argmax_firing_subtime + window - sub_window -(peak_idx[i] - subpeak_idx[i][j])
                        
                    stats[rtm + '_argmax_sub'] = sub_event_argmax_firing
                    stats[rtm + '_max_sub'] = sub_event_max_firing           
        
    for adp in adapt_array: 
        time_traces[adp] = trim_brian_monitor(network[adp],np.mean(network[adp].curr_adapt, axis=0), pA, start_time, end_time)[1]
    
    for mmp in mempo_array: 
        time_traces[mmp] = trim_brian_monitor(network[mmp],np.mean(network[mmp].v, axis=0), mV, start_time, end_time)[1]
    
    curve_t = time_traces['rtm_t']
    curve_a = time_traces['rtm_a']
    
    if n_events > 0:
        area_both = np.zeros(n_events)
        area_either = np.zeros(n_events)
        ratio = np.zeros(n_events)
        area_t_not_a = np.zeros(n_events)
        area_a_not_t = np.zeros(n_events)
        
        for i in range(n_events):
            win_start = int(peak_idx[i]- window)
            win_end = int(peak_idx[i]+ window)
            curve_t_subset = curve_t[win_start:win_end + 1]
            curve_a_subset = curve_a[win_start:win_end + 1]
            x_values = np.linspace(win_start, win_end, len(curve_t_subset))
            min_curve = np.minimum(curve_t_subset, curve_a_subset)
            area_both[i] = np.trapz(min_curve, x_values)
            area_either[i] = np.trapz(curve_t_subset, x_values) + np.trapz(curve_a_subset, x_values) - area_both[i]
            ratio[i] = area_both[i] / area_either[i] if area_either[i] != 0 else np.inf
            area_t_not_a[i] = np.trapz(curve_t_subset - min_curve, x_values)
            area_a_not_t[i] = np.trapz(curve_a_subset - min_curve, x_values)
            
        stats['area_both'] = area_both*sim_dt/second
        stats['area_either'] = area_either*sim_dt/second
        stats['ratio'] = ratio
        stats['area_t_not_a'] = area_t_not_a*sim_dt/second
        stats['area_a_not_t'] = area_a_not_t*sim_dt/second


    return stats, time_traces, test_params