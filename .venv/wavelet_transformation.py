import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import pywt  # PyWavelets library for wavelet analysis

# Parameters
CHUNK = 1024  # Number of samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (samples per second)
WAVELET = 'db4'  # Using Daubechies wavelet with 4 vanishing moments

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Set up the plot
fig, (ax_time, ax_wavelet) = plt.subplots(2, 1)

# Time domain plot (Waveform)
x_time = np.arange(0, CHUNK)  # X-axis for time domain
line_time, = ax_time.plot(x_time, np.random.rand(CHUNK))
ax_time.set_ylim(-32768, 32767)  # 16-bit audio has a range from -32768 to 32767
ax_time.set_xlim(0, CHUNK)
ax_time.set_title('Time Domain (Waveform)')

# Wavelet domain plot (Wavelet Coefficients)
x_wavelet = np.arange(0, CHUNK)
line_wavelet, = ax_wavelet.plot(x_wavelet, np.random.rand(CHUNK))
ax_wavelet.set_title('Wavelet Coefficients')
ax_wavelet.set_xlim(0, CHUNK)
ax_wavelet.set_ylim(-1000, 1000)  # Adjust range based on coefficient values

# Wavelet decomposition function
def apply_wavelet(data):
    # Perform Discrete Wavelet Transform (DWT) with PyWavelets
    coeffs = pywt.wavedec(data, WAVELET, level=5)
    # Reconstruct wavelet coefficients into a flat array for plotting
    coeff_array, _ = pywt.coeffs_to_array(coeffs)
    return coeff_array

# Function to update the plots
def update_plot():
    # Read audio data from the stream
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)

    # Update time domain plot
    line_time.set_ydata(data)

    # Apply wavelet transform and update wavelet domain plot
    wavelet_coeffs = apply_wavelet(data)
    line_wavelet.set_ydata(wavelet_coeffs[:CHUNK])  # Display the first CHUNK samples

    # Return updated plots
    return line_time, line_wavelet

# Animation update function
def animate(frame):
    return update_plot()

# Set up the animation
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, animate, interval=50, blit=True)

plt.show()

# Close the stream when done
stream.stop_stream()
stream.close()
p.terminate()
