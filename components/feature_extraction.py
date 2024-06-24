import librosa
import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectogram(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=50000)

    # Generate mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')

    plt.savefig(save_path)
    plt.close()