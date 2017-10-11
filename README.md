# Time Rescaling
Tools for evaluating the goodness of fit of a point process model.

### Dependencies ###
+  numpy
+  scipy
+  matplotlib

### References ###
1. Brown, E.N., Barbieri, R., Ventura, V., Kass, R.E., and Frank, L.M. (2002). The time-rescaling theorem and its application to neural spike train data analysis. Neural Computation 14, 325-346.
2. Wiener, M.C. (2003). An adjustment to the time-rescaling method for application to short-trial spike train data. Neural Computation 15, 2565-2576.
3. Truccolo, W. (2004). A Point Process Framework for Relating Neural Spiking Activity to Spiking History, Neural Ensemble, and Extrinsic Covariate Effects. Journal of Neurophysiology 93, 1074–1089.


### Example Usage ###

#### Fit a model
```python
import numpy as np
import matplotlib.pyplot as plt
from patsy import dmatrix
from statsmodels.api import GLM, families

def simulate_poisson_process(rate, sampling_frequency):
    '''

    Parameters
    ----------
    rate : ndarray
    sampling_frequency : float

    Returns
    -------
    poisson_point_process : ndarray
        Same shape as rate.
    '''
    return np.random.poisson(rate / sampling_frequency)

n_time, n_trials = 1500, 1000
sampling_frequency = 1500

# Firing rate starts at 5 Hz and switches to 10 Hz
firing_rate = np.ones((n_time, n_trials)) * 10
firing_rate[:n_time // 2, :] = 5
spike_train = simulate_poisson_process(
    firing_rate, sampling_frequency)
time = (np.arange(0, n_time)[:, np.newaxis] / sampling_frequency *
        np.ones((1, n_trials)))
trial_id = (np.arange(n_trials)[np.newaxis, :]
           * np.ones((n_time, 1)))

# Fit a spline model to the firing rate
design_matrix = dmatrix('bs(time, df=5)', dict(time=time.ravel()))
fit = GLM(spike_train.ravel(), design_matrix,
          family=families.Poisson()).fit()
```

#### Use time rescaling to analyze goodness of fit

```python
from .time_rescale import TimeRescaling

conditional_intensity = fit.mu
rescaled = TimeRescaling(conditional_intensity,
                         spike_train.ravel(),
                         trial_id.ravel())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
rescaled.plot_ks(ax=axes[0])
rescaled.plot_rescaled_ISI_autocorrelation(ax=axes[1])
```

![Goodness of fit, not adjusted for short trials](time_rescaling_ks_autocorrelation.png)

#### Use time rescaling to analyze goodness of fit but adjust for short trials

```python
rescaled_adjusted = TimeRescaling(conditional_intensity,
                                  spike_train.ravel(),
                                  trial_id.ravel(),
                                  adjust_for_short_trials=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
rescaled_adjusted.plot_ks(ax=axes[0])
rescaled_adjusted.plot_rescaled_ISI_autocorrelation(ax=axes[1])
```

![Goodness of fit, not adjsuted for short trials](time_rescaling_ks_autocorrelation_adjusted.png)
