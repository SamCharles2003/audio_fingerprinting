import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import wave
import struct

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


def start_record():
    global record, duration, stat
    try:
        while stat:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
            fft_data = np.fft.fft(data)
            fft_data_abs = np.abs(fft_data[:CHUNK // 2])
            fft_data_db = 20 * np.log10(fft_data_abs / np.iinfo(np.int16).max + 1e-10)
            temp = {(duration * 0.01): (fft_data_db, fft_data_abs)}
            record.update(temp)
            duration += 1

    except KeyboardInterrupt:
        stat = False
    finally:
        print("Recording stopped. Generating spectrogram...")

        # Create spectrogram
        times = np.array(list(record.keys()))
        spectrogram_data = np.array([data[0] for data in record.values()]).T

        plt.figure(figsize=(12, 8))
        plt.imshow(spectrogram_data, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), 0, RATE / 2],
                   cmap='viridis', norm=LogNorm(vmin=1e-7, vmax=1e2))

        plt.colorbar(label='Amplitude (dB)')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        plt.title('Audio Spectrogram')
        plt.tight_layout()
        plt.show()

        # Regenerate audio
        regenerate_audio()


def regenerate_audio():
    print("Regenerating audio...")
    regenerated_audio = []

    for _, (_, fft_data_abs) in record.items():
        # Create a symmetric spectrum (mirror the positive frequencies)
        full_spectrum = np.concatenate([fft_data_abs, fft_data_abs[-2:0:-1]])

        # Assume zero phase (this is where we lose accuracy)
        fft_data_complex = full_spectrum * np.exp(1j * np.zeros(len(full_spectrum)))

        # Inverse FFT to get the time domain signal
        time_signal = np.fft.ifft(fft_data_complex).real

        # Normalize and convert to 16-bit integer
        time_signal = np.int16(time_signal / np.max(np.abs(time_signal)) * 32767)

        regenerated_audio.extend(time_signal)

    # Save the regenerated audio as a WAV file
    with wave.open('regenerated_audio.wav', 'w') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 2 bytes for 16-bit audio
        wf.setframerate(RATE)
        wf.writeframes(struct.pack(f'{len(regenerated_audio)}h', *regenerated_audio))

    print("Audio regenerated and saved as 'regenerated_audio.wav'")


start_record()

# Close the stream when done
stream.stop_stream()
stream.close()
p.terminate()