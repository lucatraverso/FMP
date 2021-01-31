"""
Module: LibUNTREF.C2_Fourier
"""
import numpy as np
import librosa

def generate_matrix_dft(N,K):
    """Generate a DFT (discrete Fourier transfrom) matrix
    Args:
        N: Number of samples
        K: Number of frequency bins
    Returns:
        dft: The DFT matrix
    """
    dft = np.zeros((K,N),dtype=np.complex128)
    for n in range(N):
        for k in range(K):
            dft[k,n] = np.exp(-2j*np.pi*k*n/N)
    return dft

def dft(x):
    """Compute the discrete Fourier transfrom (DFT)
    Args:
        x: Signal to be transformed
    Returns:
        X: Fourier transform of `x`
    """
    x = x.astype(np.complex128)
    N = len(x)
    dft_mat = generate_matrix_dft(N,N)
    return np.dot(dft_mat,x)

def twiddle(N):
    """Generate twiddle factors used in the computation of the FFT
    Args:
        N: Number of samples
    Returns:
        sigma: The twiddle factors
    """
    k = np.arange(N//2)
    sigma = np.exp(-2j*np.pi*k/N)
    return sigma

def twiddle_inv(N):
    """Generate the twiddle factors used in the computation of the Inverse fast Fourier transform (IFFT)
    Args:
        N: Number of samples
    Returns:
        sigma: The twiddle factors
    """
    n = np.arange(N//2)
    sigma = np.exp(2j*np.pi*n/N)
    return sigma

def fft(x):
    """Compute the fast Fourier transform (FFT)
    Args:
        x: Signal to be transformed
    Returns:
        X: Fourier transform of `x`
    """
    x = x.astype(np.complex128)
    N = len(x)
    log2N = np.log2(N) # numero n tal que 2^n=N
    assert log2N == int(log2N), 'N debe ser potencia de 2!'
    X = np.zeros(N,dtype=np.complex128)

    if N == 1:
        return x
    else:
        this_range = np.arange(N)
        indice_par = this_range % 2 == 0 # checks if an index is even
        indice_impar = this_range % 2 == 1 # checks if an index is odd
        A = fft(x[indice_par])
        B = fft(x[indice_impar]) 
        C = twiddle(N) * B
        X[:N//2] = A + C
        X[N//2:] = A - C
        return X

def ifft_noscale(X):
    """Compute the inverse fast Fourier transform (IFFT) without the final scaling factor of 1/N
    Args:
        X: Fourier transform of `x`
    Returns:
        x: Inverse Fourier transform of `x`
    """
    X = X.astype(np.complex128)
    N = len(X)
    log2N = np.log2(N)
    assert log2N == int(log2N), 'N must be a power of two!'
    x = np.zeros(N, dtype=np.complex128)

    if N == 1:
        return X
    else:
        this_range = np.arange(N)
        A = ifft_noscale(X[this_range % 2 == 0])
        B = ifft_noscale(X[this_range % 2 == 1])
        C = twiddle_inv(N) * B
        x[:N//2] = A + C
        x[N//2:] = A - C
        return x

def ifft(X):
    """Compute the inverse fast Fourier transform (IFFT)
    Args:
        X: Fourier transform of `x`
    Returns:
        x: Inverse Fourier transform of `x
    """
    return ifft_noscale(X) / len(X)

def stft_basic(x,w,H=8,only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)
    Args:
        x: Signal to be transformed
        w: Window function
        H: Hop size
        only_positive_frequencies: Return only positive frequency part of spectrum (non-invertible)
    Returns:
        X: The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L-N)/H).astype(int)+1
    X = np.zeros((N,M),dtype='complex')
    for m in range(M):
        x_win = x[m*H:m*H+N]*w
        X_win = np.fft.fft(x_win)
        X[:,m] = X_win

    if only_positive_frequencies:
        K = 1+N//2
        X = X[0:K,:]
    return X

def stft(x,w,H=512,zero_padding=0,only_positive_frequencies=False):
    """Compute the discrete short-time Fourier transform (STFT)
    Args:
        x: Signal to be transformed
        w: Window function
        H: Hop size
        zero_padding: Number of zeros to be padded after windowing and before the Fourier transform of a frame
            (Note: The purpose of this step is to increase the frequency sampling.)
        only_positive_frequencies: Return only positive frequency part of spectrum (non-invertible)
    Returns:
        X: The discrete short-time Fourier transform
    """
    N = len(w)
    x = np.concatenate((np.zeros(N//2),x,np.zeros(N//2)))

    L = len(x)
    M = int(np.floor((L-N)/H))+1

    X = np.zeros((N+zero_padding,M),dtype=np.complex128)
    zero_padding_vector = np.zeros((zero_padding,),dtype=x.dtype)

    for m in range(M):
        x_win = x[m*H:m*H+N]*w
        if zero_padding > 0:
            x_win = np.concatenate((x_win,zero_padding_vector))
        X_win = fft(x_win)
        X[:,m] = X_win

    if only_positive_frequencies:
        K = 1+(N+zero_padding)//2
        X = X[0:K,:]
    return X

def istft(X,w,H,L,zero_padding=0):
    """Compute the inverse discrete short-time Fourier transform (ISTFT)
    Args:
        X: The discrete short-time Fourier transform
        w: Window function
        H: Hop size
        L: Length of time signal
        zero_padding: Number of zeros to be padded after windowing and before the Fourier transform of a frame
    Returns:
        x_rec: Reconstructed time signal
    """
    N = len(w)
    L = L+N
    M = X.shape[1]
    w_sum = np.zeros(L)
    x_win_sum = np.zeros(L)
    w_sum = np.zeros(L)
    for m in range(M):
        start_idx, end_idx = m*H, m*H+N+zero_padding
        if start_idx > L:
            break

        x_win = ifft(X[:,m])
        if end_idx > L:
            end_idx = L
            x_win = x_win[:end_idx-start_idx]
            cur_w = w[:end_idx-start_idx]
        else:
            cur_w = w

        # Avoid imaginary values (due to floating point arithmetic)
        x_win_real = np.real(x_win)
        x_win_sum[start_idx:end_idx] = x_win_sum[start_idx:end_idx] + x_win_real
        w_shifted = np.zeros(L)
        w_shifted[start_idx:start_idx + len(cur_w)] = cur_w
        w_sum = w_sum + w_shifted
    # Avoid division by zero
    w_sum[w_sum == 0] = np.finfo(np.float32).eps
    x_rec = x_win_sum/w_sum
    x_rec = x_rec[N//2:-N//2]
    return x_rec

def stft_conventionFMP(x, Fs, N, H, pad_mode='constant', center=True, mag=False, gamma=0):
    """Compute the discrete short-time Fourier transform (STFT)
    Args:
        x: Signal to be transformed
        Fs: Sampling rate
        N: Window size
        H: Hopsize
        pad_mode: Padding strategy is used in librosa
        center: Centric view as used in librosa
        mag: Computes magnitude STFT if mag==True
        gamma: Constant for logarithmic compression (only applied when mag==True)
    Returns:
        X: Discrete (magnitude) short-time Fourier transform
    """
    X = librosa.stft(x, n_fft=N, hop_length=H, win_length=N,
                     window='hann', pad_mode=pad_mode, center=center)
    if mag==True:
        X = np.abs(X)**2
        if gamma>0:
            X = np.log(1 + gamma * X)
    F_coef = librosa.fft_frequencies(sr=Fs, n_fft=N)
    T_coef = librosa.frames_to_time(np.arange(X.shape[1]), sr=Fs, hop_length=H)
    # T_coef = np.arange(X.shape[1]) * H/Fs
    # F_coef = np.arange(N//2+1) * Fs/N
    return X, T_coef, F_coef














