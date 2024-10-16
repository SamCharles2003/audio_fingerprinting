import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation

# Parameters
CHUNK = 1024  # Number of samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (samples per second)

# Frequency range for the spectrogram
LOW_FREQ = 20  # Lower bound in Hz
HIGH_FREQ = 5000  # Upper bound in Hz

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the plot
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)
ax_spec = fig.add_subplot(gs[0])
ax_time = fig.add_subplot(gs[1])
ax_freq = fig.add_subplot(gs[2])

# Prepare frequency array for spectrogram
freq = np.linspace(0, RATE / 2, CHUNK // 2)
freq_mask = (freq >= LOW_FREQ) & (freq <= HIGH_FREQ)
freq_plot = freq[freq_mask]

# Prepare time array for spectrogram
time_window = 100  # Number of time steps to display
time = np.linspace(0, time_window * CHUNK / RATE, time_window)

# Initialize spectrogram data
spectrogram = np.zeros((len(freq_plot), time_window))

# Create the spectrogram plot
im = ax_spec.imshow(spectrogram, aspect='auto', origin='lower',
                    extent=[time[0], time[-1], freq_plot[0], freq_plot[-1]],
                    cmap='viridis', norm=LogNorm(vmin=1e-7, vmax=1e2))

ax_spec.set_ylim(LOW_FREQ, HIGH_FREQ)
ax_spec.set_xlabel('Time (s)')
ax_spec.set_ylabel('Frequency (Hz)')
ax_spec.set_title('Real-time Audio Spectrogram')

# Add colorbar
cbar = fig.colorbar(im, ax=ax_spec)
cbar.set_label('Magnitude (dB)', rotation=270, labelpad=15)


# Time domain plot (Waveform)
x_time = np.arange(0, CHUNK)
line_time, = ax_time.plot(x_time, np.zeros(CHUNK))
ax_time.set_ylim(-32768, 32767)
ax_time.set_xlim(0, CHUNK)
ax_time.set_title('Time Domain (Waveform)')
ax_time.set_xlabel('Sample')
ax_time.set_ylabel('Amplitude')

# Frequency domain plot (FFT)
fft_x = np.fft.fftfreq(CHUNK, 1.0 / RATE)[:CHUNK // 2]
line_freq, = ax_freq.semilogx(fft_x[1:], np.zeros(CHUNK // 2 - 1))
ax_freq.set_xlim(20, RATE / 2)
ax_freq.set_ylim(0, 50)
ax_freq.set_title('Frequency Domain (FFT)')
ax_freq.set_xlabel('Frequency (Hz)')
ax_freq.set_ylabel('Magnitude (dB)')

# Apply a windowing function (Hanning)
window = np.hanning(CHUNK)

def update_plots(frame):
    global spectrogram

    # Read audio data
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

    # Update time domain plot
    line_time.set_ydata(data)

    # Apply window function
    windowed_data = data * window

    # Compute FFT and magnitude
    fft_data = np.abs(np.fft.fft(data)[:CHUNK // 2])
    fft_data_filtered = fft_data[freq_mask]

    # Convert to dB scale with reference to full scale
    fft_data_db = 20 * np.log10(fft_data / np.iinfo(np.int16).max + 1e-10)

    # Update frequency domain plot
    line_freq.set_ydata(fft_data_db[1:])

    # Update spectrogram data
    spectrogram = np.roll(spectrogram, -1, axis=1)
    spectrogram[:, -1] = fft_data_db[freq_mask]

    # Update spectrogram plot
    im.set_array(spectrogram)

    return im, line_time, line_freq

# Set up the animation
ani = FuncAnimation(fig, update_plots, interval=50, blit=True)
plt.savefig('spectrogram.png')
print("Spectrogram saved as 'spectrogram.png'")
plt.tight_layout()
plt.show()

# Close the stream when done
stream.stop_stream()
stream.close()
p.terminate()