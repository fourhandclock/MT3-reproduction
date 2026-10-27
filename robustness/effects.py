import numpy as np
from scipy.signal import butter,resample_poly,sosfiltfilt


def rms(audio):
    return float(np.sqrt(np.mean(np.square(np.asarray(audio,dtype=np.float64)))))


def apply_effect(audio,sample_rate,parameters):
    """Length-preserving, RMS-matched offline tone simulations on the full mixture.

    Zero-phase filtering and centered polyphase resampling avoid adding a fixed
    label delay. These are analytic effects, not calibrated guitar/amp models.
    """
    audio=np.asarray(audio,dtype=np.float32)
    if audio.ndim!=1: raise ValueError('Effect input must be mono.')
    if not parameters: return audio.copy()
    original_rms=rms(audio)
    if original_rms<1e-9: return audio.copy()
    output=audio.astype(np.float64)
    if 'low_db' in parameters or 'high_db' in parameters:
        split=parameters.get('split_hz',1000.0)
        low=sosfiltfilt(butter(2,split,fs=sample_rate,output='sos'),output)
        high=output-low
        output=low*10**(parameters.get('low_db',0)/20)+high*10**(parameters.get('high_db',0)/20)
    if 'drive_db' in parameters:
        oversampled=resample_poly(output,4,1)
        drive=10**(parameters['drive_db']/20)*0.15/max(rms(output),1e-9)
        oversampled=np.tanh(oversampled*drive)
        output=resample_poly(oversampled,1,4)[:len(audio)]
    if 'cabinet_hz' in parameters:
        output=sosfiltfilt(butter(2,parameters['cabinet_hz'],fs=sample_rate,output='sos'),output)
    output*=original_rms/max(rms(output),1e-9)
    peak=float(np.max(np.abs(output)))
    if peak>0.99: output*=0.99/peak
    if len(output)!=len(audio) or not np.isfinite(output).all(): raise ValueError('Invalid rendered effect.')
    return output.astype(np.float32)
