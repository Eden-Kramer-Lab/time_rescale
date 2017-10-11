'''Tools for evaluating the goodness of fit of a point process model.

References
----------
.. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank, L.M.
       (2002). The time-rescaling theorem and its application to neural
       spike train data analysis. Neural Computation 14, 325-346.
.. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method for
       application to short-trial spike train data. Neural Computation 15,
       2565-2576.

'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon
from scipy.signal import correlate


class TimeRescaling(object):
    '''Evaluates the goodness of fit of a point process model by
    transforming the fitted model into a unit rate Poisson process [1].

    Attributes
    ----------
    conditional_intensity : ndarray, shape (n_time,)
        The fitted model mean response rate at each time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    trial_id : ndarray, shape (n_time,), optional
        The label identifying time point with trial. If `trial_id` is set
        to None, then all time points are treated as part of the same
        trial. Otherwise, the data will be grouped by trial.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [2].

    References
    ----------
    .. [1] Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank,
           L.M. (2002). The time-rescaling theorem and its application to
           neural spike train data analysis. Neural Computation 14, 325-346.
    .. [2] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.

    '''
    def __init__(self, conditional_intensity, is_spike, trial_id=None,
                 adjust_for_short_trials=False):
        self.conditional_intensity = conditional_intensity
        if trial_id is None:
            trial_id = np.ones_like(conditional_intensity)
        self.trial_id = trial_id
        self.is_spike = is_spike
        self.adjust_for_short_trials = adjust_for_short_trials

    @property
    def n_spikes(self):
        '''Number of total spikes.'''
        return np.nonzero(self.is_spike)[0].size

    def uniform_rescaled_ISIs(self):
        '''Rescales the interspike intervals (ISIs) to unit rate Poisson,
        adjusts for short time intervals, and transforms the ISIs to a
        uniform distribution for easier analysis.'''

        trial_IDs = np.unique(self.trial_id)
        uniform_rescaled_ISIs_by_trial = []
        for trial in trial_IDs:
            is_trial = np.in1d(self.trial_id, trial)
            uniform_rescaled_ISIs_by_trial.append(
                uniform_rescaled_ISIs(
                    self.conditional_intensity[is_trial],
                    self.is_spike[is_trial], self.adjust_for_short_trials)
            )

        return np.concatenate(uniform_rescaled_ISIs_by_trial)

    def ks_statistic(self):
        '''Measures the maximum distance of the rescaled ISIs from the unit
        rate Poisson.

        Smaller maximum distance means better fitting model.

        '''
        uniform_cdf_values = _uniform_cdf_values(self.n_spikes)
        return ks_statistic(np.sort(self.uniform_rescaled_ISIs()),
                            uniform_cdf_values)

    def rescaled_ISI_autocorrelation(self):
        '''Examine rescaled ISI dependence.

        Should be independent if the transformation to unit rate Poisson
        process fits well.

        '''
        # Avoid -inf and inf when transforming to normal distribution.
        u = self.uniform_rescaled_ISIs()
        u[u == 0] = np.finfo(float).eps
        u[u == 1] = 1 - np.finfo(float).eps

        normal_rescaled_ISIs = norm.ppf(u)

        c = correlate(normal_rescaled_ISIs, normal_rescaled_ISIs)
        return c / c.max()

    def plot_ks(self, ax=None):
        '''Plots the rescaled ISIs versus a uniform distribution to
        examine how close the rescaled ISIs are to the unit rate Poisson.

        Parameters
        ----------
        ax : matplotlib axis handle, optional
            If None, plots on the current axis handle.

        Returns
        -------
        ax : axis_handle

        '''
        return plot_ks(self.uniform_rescaled_ISIs(), ax=ax)

    def plot_rescaled_ISI_autocorrelation(self, ax=None):
        '''Plot the rescaled ISI dependence.

        Should be independent if the transformation to unit rate Poisson
        process fits well.

        Parameters
        ----------
        ax : matplotlib axis handle, optional
            If None, plots on the current axis handle.

        Returns
        -------
        ax : axis_handle

        '''
        return plot_rescaled_ISI_autocorrelation(
            self.rescaled_ISI_autocorrelation(), ax=ax)


def _uniform_cdf_values(n_spikes):
    '''Model based cumulative distribution function values. Used for
    plotting the `uniform_rescaled_ISIs`.

    Parameters
    ----------
    n_spikes : int
        Total number of spikes.

    Returns
    -------
        uniform_cdf_values : ndarray, shape (n_spikes,)


    '''
    return (np.arange(n_spikes) + 0.5) / n_spikes


def ks_statistic(empirical_cdf, model_cdf):
    '''Compares the empirical and model-based distribution using the
    Kolmogorov-Smirnov statistic.

    Parameters
    ----------
    empirical_cdf : ndarray
    model_cdf : ndarray

    Returns
    -------
    ks_statistic : float

    '''
    try:
        return np.max(np.abs(empirical_cdf - model_cdf))
    except ValueError:
        return np.nan


def _rescaled_ISIs(integrated_conditional_intensity, is_spike):
    '''Rescales the interspike intervals (ISIs) to unit rate Poisson.

    Parameters
    ----------
    integrated_conditional_intensity : ndarray, shape (n_time,)
        The cumulative conditional_intensity integrated over time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.

    Returns
    -------
    rescaled_ISIs : ndarray, shape (n_spikes,)

    '''
    ici_at_spike = integrated_conditional_intensity[is_spike.nonzero()]
    ici_at_spike = np.concatenate((np.array([0]), ici_at_spike))
    return np.diff(ici_at_spike)


def _max_transformed_interval(integrated_conditional_intensity,
                              is_spike, rescaled_ISIs):
    '''Weights for each time in censored trials.

    Parameters
    ----------
    integrated_conditional_intensity : ndarray, shape (n_time,)
        The cumulative conditional_intensity integrated over time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    rescaled_ISIs : ndarray, shape (n_spikes,)

    Returns
    -------
    max_transformed_interval : ndarray, shape (n_spikes,)

    '''
    ici_at_spike = integrated_conditional_intensity[is_spike.nonzero()]
    return (integrated_conditional_intensity[-1] - ici_at_spike +
            rescaled_ISIs)


def uniform_rescaled_ISIs(conditional_intensity, is_spike,
                          adjust_for_short_trials=True):
    '''Rescales the interspike intervals (ISIs) to unit rate Poisson,
    adjusts for short time intervals, and transforms the ISIs to a
    uniform distribution for easier analysis.

    Parameters
    ----------
    conditional_intensity : ndarray, shape (n_time,)
        The fitted model mean response rate at each time.
    is_spike : bool ndarray, shape (n_time,)
        Whether or not the neuron has spiked at that time.
    adjust_for_short_trials : bool, optional
        If the trials are short and neuron does not spike often, then
        the interspike intervals can be longer than the trial. In this
        situation, the interspike interval is censored. If
        `adjust_for_short_trials` is True, we take this censoring into
        account using the adjustment in [1].

    Returns
    -------
    uniform_rescaled_ISIs : ndarray, shape (n_spikes,)

    References
    ----------
    .. [1] Wiener, M.C. (2003). An adjustment to the time-rescaling method
           for application to short-trial spike train data. Neural
           Computation 15, 2565-2576.

    '''
    integrated_conditional_intensity = np.cumsum(conditional_intensity)
    rescaled_ISIs = _rescaled_ISIs(
        integrated_conditional_intensity, is_spike)

    if adjust_for_short_trials:
        max_transformed_interval = expon.cdf(_max_transformed_interval(
            integrated_conditional_intensity, is_spike, rescaled_ISIs))
    else:
        max_transformed_interval = 1

    return expon.cdf(rescaled_ISIs) / max_transformed_interval


def plot_ks(uniform_rescaled_ISIs, ax=None):
    n_spikes = uniform_rescaled_ISIs.size
    uniform_cdf_values = _uniform_cdf_values(n_spikes)
    uniform_rescaled_ISIs = np.sort(uniform_rescaled_ISIs)

    ci = 1.36 / np.sqrt(n_spikes)

    if ax is None:
        ax = plt.gca()
    ax.plot(uniform_cdf_values, uniform_cdf_values - ci,
            linestyle='--', color='red')
    ax.plot(uniform_cdf_values, uniform_cdf_values + ci,
            linestyle='--', color='red')
    ax.scatter(uniform_rescaled_ISIs, uniform_cdf_values)

    ax.set_xlabel('Empirical CDF')
    ax.set_ylabel('Expected CDF')

    return ax


def plot_rescaled_ISI_autocorrelation(rescaled_ISI_autocorrelation,
                                      ax=None):
    n_spikes = rescaled_ISI_autocorrelation.size // 2 + 1
    lag = np.arange(-n_spikes + 1, n_spikes)
    if ax is None:
        ax = plt.gca()
    ci = 1.96 / np.sqrt(n_spikes)
    ax.scatter(lag, rescaled_ISI_autocorrelation)
    ax.axhline(ci, linestyle='--', color='red')
    ax.axhline(-ci, linestyle='--', color='red')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')

    return ax
