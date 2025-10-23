import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def analyze_audio(file_path, sr=None):
    samples, sample_rate = librosa.load(file_path, sr=sr)

    # RMS and ZCR
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=samples, frame_length=frame_length, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=samples, frame_length=frame_length, hop_length=hop_length)[0]

    # Time
    frames = np.arange(len(rms))
    t_rms = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length)
    t_zcr = t_rms

    # Voiced/unvoiced threshold
    threshold_rms = 0.2 * np.max(rms)
    voiced_flags = rms > threshold_rms

    # Plot
    plt.figure(figsize=(14, 10))

    # Waveform
    plt.subplot(3,1,1)
    librosa.display.waveshow(samples, sr=sample_rate, alpha=0.5)
    plt.title("Original Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Highlight voiced/unvoiced
    prev_flag = None
    start_time = 0
    for i, flag in enumerate(voiced_flags):
        if prev_flag is None:
            prev_flag = flag
            start_time = t_rms[i]
        elif prev_flag != flag:
            end_time = t_rms[i]
            color = 'red' if prev_flag else 'blue'
            plt.axvspan(start_time, end_time, color=color, alpha=0.3)
            prev_flag = flag
            start_time = t_rms[i]
    if prev_flag is not None:
        end_time = t_rms[-1]
        color = 'red' if prev_flag else 'blue'
        plt.axvspan(start_time, end_time, color=color, alpha=0.3)

    # RMS
    plt.subplot(3,1,2)
    plt.plot(t_rms, rms, color='orange')
    plt.title("RMS Energy")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")

    # ZCR
    plt.subplot(3,1,3)
    plt.plot(t_zcr, zcr, color='green')
    plt.title("Zero-Crossing Rate (ZCR)")
    plt.xlabel("Time (s)")
    plt.ylabel("ZCR")

    plt.tight_layout()
    plt.show()

    print("Sampling Rate:", sample_rate)

# Example usage
analyze_audio("/content/I am Happy.mp3")
