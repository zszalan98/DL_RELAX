import torch
import torchaudio as ta
from torchaudio.transforms import Resample, Spectrogram, AmplitudeToDB, InverseSpectrogram

import sounddevice as sd
import numpy as np


# Utility functions
def read_audio(audio_path: str):
    audio, sample_rate = ta.load(audio_path)
    # Return audio as torch.Tensor and sample rate
    return audio, sample_rate

def resample_audio(audio: torch.Tensor, sr: int, new_sr: int = 16000):
    # Resample the audio file
    if sr != new_sr:
        resampler = Resample(orig_freq=sr, new_freq=new_sr)
        new_audio = resampler(audio)
        return new_audio
    else:
        return audio

def play_audio(audio: torch.Tensor, sample_rate: int):
    audio_np = np.array(audio)
    # Play the audio
    sd.play(audio_np.squeeze(), sample_rate)


def get_spectrogram(audio: torch.Tensor, n_ftt: int = 2048, hop_length: int = 882):
    # Using torchaudio to create the spectrogram
    spec_transform = Spectrogram(n_fft=n_ftt, hop_length=hop_length)
    return spec_transform(audio)

def convert_to_db(spec: torch.Tensor):
    # Rescale spectrograms to Db scale
    return AmplitudeToDB()(spec)


def get_complex_spectrogram(audio: torch.Tensor, n_ftt: int = 2048, hop_length: int = 882):
    # Using torchaudio to create the spectrogram
    cplx_spec_transform = Spectrogram(n_fft=n_ftt, hop_length=hop_length, power=None)
    return cplx_spec_transform(audio)

def inverse_complex_spectrogram(spec: torch.Tensor, n_ftt: int = 2048, hop_length: int = 882):
    # Convert the spectrogram back to audio
    inv_spec_transform = InverseSpectrogram(n_fft=n_ftt, hop_length=hop_length)
    return inv_spec_transform(spec)


# Batched relax functions
def prepare_audio(f, play=False):
    """
    Prepare audio file for analysis.

    Args:
        f (str): File path of the audio file.
        plot (bool, optional): Whether to plot the spectrogram. Defaults to False.

    Returns:
        resampled audio signal, complex spectrogram, spectrogram, and shape of the spectrograms
    """
    # Read audio
    audio, sr = read_audio(f)

    # Resample audio to 16kHz to save compute time
    audio = resample_audio(audio, sr, 16000)

    # Get complex spectrogram
    n_ftt = 2048
    cpx_spec = get_complex_spectrogram(audio, n_ftt=n_ftt)

    # Get real spectrogram
    spec = get_spectrogram(audio, n_ftt=n_ftt)
    spec_db = convert_to_db(spec)

    # Play audio
    if play:
        play_audio(audio, sr)

    # Return
    return audio, cpx_spec, spec_db, spec.shape[1:3]
    