"""
Module: LibUNTREF.C1S3_AudioRep
"""
import numpy as np

def generate_sinusoid(dur=5, Fs=1000, amp=1, freq=1, phase=0):
    """Generation of sinusoid
    Args:
        dur: Duration (in seconds)
        Fs: Sampling rate
        amp: Amplitude of sinusoid
        freq: Frequency of sinusoid
        phase: Phase of sinusoid
    Returns:
        x: Signal
        t: Time axis (in seconds)
    """
    num_samples = int(Fs * dur)
    t = np.arange(num_samples) / Fs
    x = amp * np.sin(2*np.pi*(freq*t-phase))
    return x, t

