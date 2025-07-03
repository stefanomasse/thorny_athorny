from __future__ import division, print_function
from brian2 import *
from scipy import signal
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
