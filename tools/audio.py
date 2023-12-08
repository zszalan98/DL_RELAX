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

def save_audio(audio: torch.Tensor, sample_rate: int, save_path: str):
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    ta.save(save_path, audio, sample_rate)

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


def get_spectrogram(audio: torch.Tensor, n_ftt: int = 512, win_len: int = 500, hop_len: int = 400):
    # Using torchaudio to create the spectrogram
    spec_transform = Spectrogram(n_fft=n_ftt, win_length=win_len, hop_length=hop_len)
    return spec_transform(audio)

def convert_to_db(spec: torch.Tensor):
    # Rescale spectrograms to Db scale
    return AmplitudeToDB(stype="amplitude", top_db=80)(spec)


def get_complex_spectrogram(audio: torch.Tensor, n_ftt: int = 512, win_len: int = 500, hop_len: int = 400):
    # Using torchaudio to create the spectrogram
    cplx_spec_transform = Spectrogram(n_fft=n_ftt, win_length=win_len, hop_length=hop_len, power=None)
    return cplx_spec_transform(audio)

def inverse_complex_spectrogram(spec: torch.Tensor, n_ftt: int = 512, win_len: int = 500, hop_len: int = 400):
    # Convert the spectrogram back to audio
    inv_spec_transform = InverseSpectrogram(n_fft=n_ftt, win_length=win_len, hop_length=hop_len)
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
    cpx_spec = get_complex_spectrogram(audio)

    # Get real spectrogram
    spec = get_spectrogram(audio)
    spec_db = convert_to_db(spec)

    # Get spectrogram shape
    assert cpx_spec.shape == spec_db.shape, "The shape of the spectrograms does not match"
    spec_shape = tuple(spec_db.shape[1:3]) if len(spec_db.shape) == 3 else tuple(spec_db.shape[0:2])

    # Play audio
    if play:
        play_audio(audio, sr)

    # Return
    return audio, cpx_spec, spec_db, tuple(spec.shape[1:3])
    