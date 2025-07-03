from __future__ import division, print_function
from brian2 import *
import os
import glob
from scipy import signal
import scipy.optimize
import numpy as np



class Connectivity:
    def __init__(self, p_ij, n_j, n_i, name):
        """
        create connectivity matrix object
        Args:
            p_ij: probability of J->I connection
            n_j: number of J cells
            n_i: number of I cells
            name: name of connection
        """

        self.p_ij = p_ij
        self.n_j = int(n_j)
        self.n_i = int(n_i)
        self.name = name
        self.pre_index = 0      # indices of presynaptic neurons
        self.post_index = 0     # indices of postsynaptic neurons

        # if presynaptic input is high enough:
        if np.sqrt(self.p_ij * self.n_j) > 2:
            conn_matrix = np.random.binomial(1, self.p_ij, size=(self.n_j, self.n_i))
            self.pre_index, self.post_index = np.nonzero(conn_matrix)
        else:
            print('sqrt(p_ij * N_j) = %.2f' % np.sqrt(self.p_ij * self.n_j))
            raise ValueError('Too few presynaptic connections in %s' % self.name)
            
            
            



def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',kpsh=False, valley=False, show=False, ax=None):

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indexes of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                              & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indexes by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def calc_unit_rates(spike_monitor, t_start, t_stop):
    """
    calculate individual firing rate for each neuron

    Args:
        spike_monitor: brian spike monitor object
        t_start: calculation start time
        t_stop: calculation stop time

    Returns:
        mean_rate: mean firing rate across all neurons
        std_rate: standard deviation of firing rate across all neurons

    """

    spike_trains = spike_monitor.spike_trains()
    num_neurons = len(spike_trains)
    num_spikes = np.zeros(num_neurons)

    for i in range(num_neurons):
        num_spikes[i] = len((spike_trains[i])[(spike_trains[i] >= t_start) &
                                              (spike_trains[i] <= t_stop)])

    rates = num_spikes / (t_stop - t_start)

    mean_rate = np.mean(rates)/Hz
    std_rate = np.std(rates)/Hz

    return mean_rate, std_rate


def lorentzian(x_array, a, mu, sigma):
    """
    the lorentzian function.

    Args:
        x_array: argument of function
        a: parameter 1
        mu: parameter 2, peak center
        sigma: parameter 3, width at half maximum

    Returns:
        output: the lorentzian function
    """

    output = (a / pi) * sigma / ((x_array - mu) ** 2 + sigma ** 2)
    return output


def get_q_factor(a, sigma):
    """
    calculate Q-factor of a lorentzian

    Args:
        a: parameter 1 of Lorentzian function
        sigma: parameter 3 of Lorentzian function

    Returns:
        q_factor
    """

    peak = a / (pi * sigma)
    fwhm = 2 * sigma
    q_factor = peak / fwhm

    return q_factor


def calc_network_frequency(pop_rate, sim_time, dt,min_freq, max_freq, fit=True):
    """
    calculate Power Spectral Density (PSD) of population activity
    and try to fit it to a lorentzian function

    Args:
        pop_rate: population rate signal
        sim_time: total time of pop_rate signal
        dt: time step of pop_rate signal
        max_freq: maximum frequency of the power spectrum
        fit: [true/false] if true, try to fit PSD to lorentzian

    Returns:
        fft_freq: frequency array of FFT
        fft_psd: PSD array of FFT
        fit_params: estimated parameters of lorentzian fit
    """

    # Power Spectral Density (PSD) (absolute value of Fast Fourier Transform) centered around the mean:
    fft_psd = np.abs(np.fft.fft(pop_rate - np.mean(pop_rate)) * dt) ** 2 / (sim_time / second)

    # frequency arrays for PSD:
    fft_freq = np.fft.fftfreq(pop_rate.size, dt)

    # delete second (mirrored) half of PSD:
    fft_psd = fft_psd[:int(fft_psd.size / 2)]
    fft_freq = fft_freq[:int(fft_freq.size / 2)]

    # find argument where frequency is closest to limits:
    arg_min = (np.abs(fft_freq - min_freq)).argmin()
    arg_lim = (np.abs(fft_freq - max_freq)).argmin()

    # select power spectrum range [1,max_freq] Hz:
    fft_freq = fft_freq[arg_min:arg_lim + 1]
    fft_psd = fft_psd[arg_min:arg_lim + 1]

    fit_params = []
    if fit and (fft_psd != 0).any():

        # find maximum or power spectrum:
        i_arg_max = np.argmax(fft_psd)
        freq_max = fft_freq[i_arg_max]
        psd_max = fft_psd[i_arg_max]

        # fit power spectrum peak to Lorentzian function:
        try:
            fit_params, _ = scipy.optimize.curve_fit(lorentzian, fft_freq, fft_psd,
                                                     p0=(psd_max, freq_max, 1), maxfev=1000)
        finally:
            if fit_params is []:
                print("WARNING: Couldn't fit PSD to Lorentzian")

    return fft_freq, fft_psd, fit_params


def calc_isi_cv(spike_monitor, t_start, t_stop):
    """
    calculate the Coefficient of Variation (CV) of
    the Inter-Spike-Interval (ISI) for each neuron

    Args:
        spike_monitor: brian spike monitor
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        mean_isi_cv: mean ISI CV across all neurons
        std_isi_cv: std of ISI CV across all neurons
        check_enough_spikes: [true/false] true if there are enough spikes for calculation

    """

    mean_isi_cv = 0
    std_isi_cv = 0

    # get spike times for whole network:
    spike_trains = spike_monitor.spike_trains()

    # trim spikes within calculation time range:
    num_neurons = len(spike_trains)
    cut_spike_trains = {new_list: [] for new_list in range(num_neurons)}
    for i in range(num_neurons):
        cut_spike_trains[i] = (spike_trains[i])[(spike_trains[i] >= t_start) &
                                                (spike_trains[i] <= t_stop)]

    # check if at least one neuron spikes at least twice in the selected interval:
    n = 0
    for i in range(num_neurons):
        if (cut_spike_trains[i]).size >= 2:
            n += 1

    # n is the number of neurons that spiked at least twice.
    # if there is at least one of those:
    if n > 0:
        check_enough_spikes = True

        # calculate ISI CV for each neuron:
        all_isi_cvs = np.zeros(n)
        j = 0
        for i in range(num_neurons):
            # if the neuron spiked at least twice:
            if (cut_spike_trains[i]).size >= 2:
                # get array of ISIs:
                isi = np.diff(cut_spike_trains[i])
                # calculate average and std of ISIs:
                avg_isi = np.mean(isi)
                std_isi = np.std(isi)
                # store value of neuron ISI CV
                all_isi_cvs[j] = std_isi / avg_isi
                j = j + 1

        # calculate mean and std of all ISI CVs:
        mean_isi_cv = np.mean(all_isi_cvs)
        std_isi_cv = np.std(all_isi_cvs)

    # if not enough spikes to perform calculation:
    else:
        check_enough_spikes = False

    return mean_isi_cv, std_isi_cv, check_enough_spikes


def check_network_state(state, lfp_thres, lfp_time, lowpass_lfp, t_start, t_stop):
    """
    check if network is in the required state (non-SWR or SWR)

    Args:
        state: ['swr'/'nswr'] state to be checked
        lfp_thres: threshold that separates states
        lfp_time: time array of LFP
        lowpass_lfp: lowpass-filtered LFP trace
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        check: [true/false]

    """

    trimmed_trace = lowpass_lfp[(lfp_time > t_start) & (lfp_time < t_stop)]

    check = True
    if (state == 'nswr') and (trimmed_trace > lfp_thres).any():
        check = False

    if (state == 'swr') and (trimmed_trace < lfp_thres).any():
        check = False

    return check


def check_brian_monitor(network, mon_name, mon_attr):
    """
    check if a given brian monitor with
    a given attribute (variable being measured) exists in the network object

    Args:
        network: brian network object
        mon_name: name of monitor to check
        mon_attr: name of monitor attribute to check

    Returns:
        check: [true/false]
    """

    check = False
    if mon_name in network:
        monitor = network[mon_name]
        if hasattr(monitor, mon_attr):
            check = True

    return check


def trim_brian_monitor(monitor, attr, attr_unit, t_start, t_stop):
    """
    trim a given brian monitor attribute to a given time range

    Args:
        monitor: brian monitor object
        attr: attribute of monitor (variable being measured)
        attr_unit: output unit of attribute
        t_start: start time of calculation
        t_stop: stop time of calculation

    Returns:
        time_array: trimmed time array
        attr_array: trimmed attribute array (unit-less)
    """

    time_array = np.array(monitor.t / second)[(monitor.t >= t_start) &
                                              (monitor.t <= t_stop)]

    attr_array = np.array(attr / attr_unit)[(monitor.t >= t_start) &
                                            (monitor.t <= t_stop)]

    return time_array, attr_array


# noinspection PyTupleAssignmentBalance
def calc_low_pass_filter(data, cutoff, dt):
    """
    calculate low pass filter

    Args:
        data: input signal
        cutoff: cutoff frequency of low pass filter
        dt: time step of data signal

    Returns:
        low_pass_trace: low-pass-filtered signal
    """

    nyquist_freq = (1/2) * (1/dt)
    b_butter, a_butter = signal.butter(2, cutoff/nyquist_freq, btype='low', analog=False, output='ba')

    low_pass_trace = signal.filtfilt(b_butter, a_butter, data)

    return low_pass_trace


def calc_band_pass_filter(data, lowcut, highcut, dt):

    nyquist_freq = (1/2) * (1/dt)
    b_butter, a_butter = signal.butter(2, [lowcut/nyquist_freq, highcut/nyquist_freq], btype='band', analog=False, output='ba')

    band_pass_trace = signal.filtfilt(b_butter, a_butter, data)

    return band_pass_trace


def get_newest_file(file_name):
    """
    get the most recent file matching the input file name

    Args:
        file_name: input file name to match

    Returns:
        newest_file: most recent file matching the file_name
    """

    curr_dir = os.getcwd()
    files = glob.glob(curr_dir + '\\' + file_name)
    if len(files) > 0:
        newest_file = max(files, key=os.path.getctime)
    else:
        newest_file = ''

    return newest_file


def test_synchronicity():
    # TODO create function that tests if network is synchronous/asynchronous
    pass


def test_isi_cv():
    # TODO create function that tests if ISI CV has regular or irregular firing
    pass
