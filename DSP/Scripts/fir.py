import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def plot_zplane(h, ax):
    zeros = np.roots(h)
    ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zeros')
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='dashed', linewidth=1.5)
    ax.add_patch(circle)
    ax.axhline(0, color='black', linewidth=0.75, linestyle='dashed')
    ax.axvline(0, color='black', linewidth=0.75, linestyle='dashed')
    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_title("Z-Plane", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='dotted')

# Constants
_seg = float(1)
_Hz = float(1/_seg)
_kHz = _Hz * (10**3)

# System parameters
fs = 1 * _kHz  # Sampling frequency
Ts = 1/fs  # Sampling period

# Signal parameters
fo = 100 * _Hz  # Signal frequency
A = 1  # Signal amplitude

# Buffer length
N = 256

# Time vector
t = np.arange(0, N*Ts, Ts)
x_t = A * np.sin(2 * np.pi * fo * t) + 0.5 * np.random.randn(len(t))

# FIR Filter Design
num_taps = 11  # Number of filter taps (order+1)
fc = 150 * _Hz  # Cutoff frequency
cutoff = fc / (fs / 2)  # Normalized cutoff frequency
b = signal.firwin(num_taps, cutoff=cutoff, window='hamming', fs=fs)

# Print FIR transfer function
print("FIR Filter Transfer Function:")
print(f"Numerator Coefficients (b): {b}")
print(f"Filter Order: {num_taps - 1}")
print(f"Cutoff Frequency: {fc} Hz")
print(f"Sampling Frequency: {fs} Hz")

# Apply Filter
y_t = signal.lfilter(b, 1, x_t)

# Compute Spectrum
X_f = np.abs(fft(x_t))[:N//2]
Y_f = np.abs(fft(y_t))[:N//2]
freqs = np.linspace(0, fs/2, N//2)

# Create subplots with the same size
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot Z-Plane
plot_zplane(b, axes[0, 0])

# Plot Frequency Response
w, h = signal.freqz(b, worN=8000, fs=fs)
axes[0, 1].plot(w/_kHz, 20 * np.log10(abs(h)), color='red', linewidth=2)
axes[0, 1].set_title("FIR Frequency Response", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Frequency (KHz)", fontsize=12)
axes[0, 1].set_ylabel("Magnitude (dB)", fontsize=12)
axes[0, 1].grid(True, linestyle='dotted')

# Plot Original and Filtered Signal
axes[1, 0].plot(t, x_t, label="Original Signal", alpha=0.5, color='gray')
axes[1, 0].plot(t, y_t, label="Filtered Signal", linewidth=2, color='blue')
axes[1, 0].set_title("Signal Filtering", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Time (s)", fontsize=12)
axes[1, 0].set_ylabel("Amplitude", fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle='dotted')

# Plot Spectrum
axes[1, 1].plot(freqs/_kHz, X_f, label="Original Spectrum", alpha=0.5, color='gray')
axes[1, 1].plot(freqs/_kHz, Y_f, label="Filtered Spectrum", linewidth=2, color='blue')
axes[1, 1].set_title("Signal Spectrum", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Frequency (KHz)", fontsize=12)
axes[1, 1].set_ylabel("Magnitude", fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, linestyle='dotted')

# Adjust layout and show plots
plt.tight_layout()
plt.show()
