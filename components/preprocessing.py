import librosa
import soundfile as sf

def load_trim_audio(path,sr):
    y, sr = librosa.load(path,sr=sr)
    y = y[:25000]
    return y, sr

def save_audio(y, sr, path):
    sf.write(path,y,sr)