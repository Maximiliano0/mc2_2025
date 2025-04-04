import numpy as np
import matplotlib.pyplot as plt

# Units and constants
_seg = float(1)
_mseg = _seg * (10**-3)
_Hz = float(1/_seg)
_kHz = _Hz * (10**3)

# System parameters
fs = 1 * _kHz # Sampling frequency
Ts = 1/fs # Sampling period

# Signal parameters
fo = 10 * _Hz # Signal frequency
To = 1/fo # Signal period
A = 1 # Signal amplitude

# Buffer length
N = To/Ts

# Time vector
t = np.arange(0, N*Ts, Ts)
x_t = A * np.sin(2 * np.pi * fo * t) + np.sin(2 * np.pi * (5*fo) * t)

# Windowing
# Blackman window
w = np.blackman(len(t))
x_wt = x_t * w
# Hamming window
#w = np.hamming(len(t))
#x_wt = x_t * w
# Hanning window
#w = np.hanning(len(t))
#x_wt = x_t * w
# Bartlett window
#w = np.bartlett(len(t))
#x_wt = x_t * w
# Rectangular window
#w = np.ones(len(t))
#x_wt = x_t * w

# Create a figure with adjusted size for better readability
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# Plot the sampled signal
axes[0].stem(t/_mseg, x_t)
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Sampled Polytone Signal')
axes[0].grid()

# Plot the windowed signal
axes[1].stem(t/_mseg, x_wt)
axes[1].set_ylabel('Amplitude')
axes[1].set_title('Windowed Polytone Signal')
axes[1].grid()

# Plot the window function
axes[2].stem(t/_mseg, w)
axes[2].set_xlabel('Time [ms]')
axes[2].set_ylabel('Amplitude')
axes[2].set_title('Window')
axes[2].grid()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()

# FFT of the signal
X_f = np.fft.fft(x_t)
abs_X_f = np.abs(X_f)
f = np.arange(0, fs/2, fs/N)

X_wf = np.fft.fft(x_wt)
abs_X_wf = np.abs(X_wf)
f = np.arange(0, fs/2, fs/N)

# FFT of the window function
W_f = np.fft.fft(w)
abs_W_f = np.abs(W_f)

# Create a figure with adjusted size for better readability
fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# Plot the FFT of the signal
axes[0].stem(f/_kHz, (1/N) * abs_X_f[:int((N+1)/2)])
axes[0].set_ylabel('Magnitude')
axes[0].set_title('FFT of the Signal')
axes[0].grid()

# Plot the FFT of the windowed signal
axes[1].stem(f/_kHz, (1/N) * abs_X_wf[:int((N+1)/2)])
axes[1].set_ylabel('Magnitude')
axes[1].set_title('FFT of the Windowed Polytone Signal')
axes[1].grid()

# Plot the FFT of the window function
axes[2].stem(f/_kHz, (1/N) * abs_W_f[:int((N+1)/2)])
axes[2].set_xlabel('Frequency [kHz]')
axes[2].set_ylabel('Magnitude')
axes[2].set_title('FFT of the Window')
axes[2].grid()

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()