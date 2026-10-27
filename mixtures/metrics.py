import numpy as np
from scipy.signal import correlate, correlation_lags


def separation_metrics(target, estimate, sample_rate):
    """Same-length waveform scores, with no alignment or gain correction applied to audio."""
    target = np.asarray(target,dtype=np.float64); estimate = np.asarray(estimate,dtype=np.float64)
    if target.shape != estimate.shape or target.ndim != 1 or not target.size: raise ValueError('Waveforms must have equal nonempty mono lengths.')
    if sample_rate <= 0: raise ValueError('Sample rate must be positive.')
    if not np.isfinite(target).all() or not np.isfinite(estimate).all(): raise ValueError('Nonfinite metric inputs.')
    target = target-target.mean(); estimate = estimate-estimate.mean()
    energy = float(target@target)
    if energy < 1e-16: return {'si_sdr_db':None,'sdr_db':None,'normalized_correlation':None,'diagnostic_lag_ms':None}
    alpha = float(target@estimate)/energy
    projection = alpha*target; noise = estimate-projection
    epsilon = 1e-16
    estimate_silent = float(estimate@estimate) < epsilon
    si_sdr = None if estimate_silent else float(10*np.log10((float(projection@projection)+epsilon)/(float(noise@noise)+epsilon)))
    sdr = 10*np.log10((energy+epsilon)/(float((estimate-target)@(estimate-target))+epsilon))
    denominator = np.sqrt(energy*float(estimate@estimate))
    correlation = float(target@estimate)/denominator if denominator > epsilon and not estimate_silent else None
    # Bounded +/-100ms diagnostic only; never shift the stem or its note labels.
    values = correlate(estimate,target,mode='full',method='fft')
    lags = correlation_lags(len(estimate),len(target),mode='full')
    keep = abs(lags) <= round(.1*sample_rate)
    lag = int(lags[keep][np.argmax(abs(values[keep]))]) if float(estimate@estimate)>epsilon else None
    return {'si_sdr_db':si_sdr,'estimate_silent':estimate_silent,'sdr_db':float(sdr),'normalized_correlation':correlation,
            'diagnostic_lag_ms':1000*lag/sample_rate if lag is not None else None,'gain_projection':alpha,
            'alignment_correction_applied':False,'definition':'Mean-centered SI-SDR and direct target-error SDR; no BSS Eval filtering.'}
