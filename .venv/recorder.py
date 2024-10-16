import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Parameters
CHUNK = 1024  # Number of samples per frame
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1  # Mono audio
RATE = 44100  # Sampling rate (samples per second)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

record = {}
duration = 0
stat = True

def make_spectrogram(record):
    times = np.array(list(record.keys()))
    spectrogram_data = np.array(list(record.values())).T

    plt.figure(figsize=(12, 8))
    plt.imshow(spectrogram_data, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), 0, RATE / 2],
               cmap='viridis', norm=LogNorm(vmin=1e-7, vmax=1e2))

    plt.colorbar(label='Amplitude (dB)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Katadam Katidum')
    plt.savefig('Katadam Katidum.png')
    plt.tight_layout()
    plt.show()

def start_record():
    global record, duration, stat
    try:
        while stat:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            fft_data = np.abs(np.fft.fft(data)[:CHUNK // 2])
            fft_data_db = 20 * np.log10(fft_data / np.iinfo(np.int16).max + 1e-10)
            temp = {(duration * 0.01): fft_data_db}
            record.update(temp)
            duration += 1

    except KeyboardInterrupt:
        stat = False
    finally:
        print("Recording stopped. Generating spectrogram...")
        make_spectrogram(record)
        # Create spectrogram
start_record()

# Close the stream when done
stream.stop_stream()
stream.close()
p.terminate()