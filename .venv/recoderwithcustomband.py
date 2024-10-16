import pyaudio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import wave
import struct
import threading

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
    while stat:
        data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        fft_data = np.fft.fft(data)
        fft_data_abs = np.abs(fft_data[:CHUNK // 2])
        fft_data_db = 20 * np.log10(fft_data_abs / np.iinfo(np.int16).max + 1e-10)
        temp = {(duration * 0.01): (fft_data_db, fft_data_abs)}
        record.update(temp)
        duration += 1

def user_input():
    global stat
    while True:
        if input().lower() == 'exit':
            stat = False
            break

def generate_spectrogram():
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

def regenerate_audio(min_freq, max_freq):
    print(f"Regenerating audio within frequency band {min_freq}-{max_freq} Hz...")
    regenerated_audio = []

    for _, (_, fft_data_abs) in record.items():
        # Apply frequency band filter
        freq_resolution = RATE / CHUNK
        min_index = int(min_freq / freq_resolution)
        max_index = int(max_freq / freq_resolution)
        filtered_spectrum = np.zeros_like(fft_data_abs)
        filtered_spectrum[min_index:max_index] = fft_data_abs[min_index:max_index]

        # Create a symmetric spectrum
        full_spectrum = np.concatenate([filtered_spectrum, filtered_spectrum[-2:0:-1]])

        # Assume zero phase
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

# Start recording in a separate thread
record_thread = threading.Thread(target=start_record)
record_thread.start()

print("Recording started. Type 'exit' to stop recording.")
user_input()

# Wait for the recording thread to finish
record_thread.join()

# Close the stream
stream.stop_stream()
stream.close()
p.terminate()

print("Recording stopped. Generating spectrogram...")
generate_spectrogram()

# Save the recorder array as .npy file
np.save('recorder_data.npy', record)
print("Recorder data saved as 'recorder_data.npy'")


while True:
    user_choice = input("Enter 'band' to specify a new frequency band, or 'exit' to quit: ").lower()

    if user_choice == 'exit':
        break
    elif user_choice == 'band':
        while True:
            try:
                min_freq = float(input("Enter minimum frequency (Hz): "))
                max_freq = float(input("Enter maximum frequency (Hz): "))
                if 0 <= min_freq < max_freq <= RATE / 2:
                    regenerate_audio(min_freq, max_freq)
                    break
                else:
                    print(f"Invalid range. Please enter values between 0 and {RATE / 2} Hz.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
    else:
        print("Invalid input. Please enter 'band' or 'exit'.")

print("Program ended. Thank you for using the audio regeneration tool!")