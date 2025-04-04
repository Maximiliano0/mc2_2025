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
x_t = A * np.sin(2 * np.pi * fo * t)

# Plot the sampled signal
plt.stem(t/_mseg, x_t)
plt.xlabel('Time [ms]')
plt.ylabel('Amplitude')
plt.title('Sampled Sinusoidal Signal')
plt.grid()
plt.show()

# FFT of the signal
X_f = np.fft.fft(x_t)
abs_X_f = np.abs(X_f)
f = np.arange(0, fs/2, fs/N)

# Plot the FFT of the signal
plt.stem(f/_kHz, (1/N) * abs_X_f[:int((N+1)/2)])
plt.xlabel('Frequency [kHz]')
plt.ylabel('Magnitude')
plt.title('FFT of the Signal')
plt.grid()
plt.show()