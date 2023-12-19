import numpy as np
import matplotlib.pyplot as plt
import librosa
import torch
from torchaudio.transforms import Resample, Spectrogram, AmplitudeToDB, InverseSpectrogram
import soundfile as sf
from tools.audio import inverse_complex_spectrogram, convert_to_db, get_spectrogram
import os




def filter_complex_spectrogram(spectrogram, importance, low_limit):
    # Get the magnitude and phase of the complex spectrogram
    magnitude = np.abs(spectrogram)
    phase = np.angle(spectrogram)

    # Normalize the magnitude array between 1 and low_limit
    normalized1_0 = (importance - torch.min(importance)) / (torch.max(importance) - torch.min(importance))
    normalized = normalized1_0 * (1 - low_limit) + low_limit

    # Apply filter on the magnitude
    filtered_magnitude = ((magnitude - torch.min(magnitude)) * normalized + torch.min(magnitude))

    # Combine the filtered magnitude with the original phase to get the filtered complex spectrogram
    filtered_spectrogram = filtered_magnitude * np.exp(1j * phase)

    return filtered_spectrogram, normalized

def get_reconstructed_audios(spectrogram, importance, filtered, normalized, non_filtered):

    # Inverse operation: Reconstruct audio from complex spectrogram
    original_audio = inverse_complex_spectrogram(non_filtered)
    filtered_audio = inverse_complex_spectrogram(filtered)

    # DEBUGING  
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Original spectrogram
    fig.colorbar(ax1.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower'), ax=ax1)
    ax1.set_title('Spectrogram of the original audio')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')

    # Importance
    fig.colorbar(ax2.imshow(importance, aspect='auto', cmap='viridis', origin='lower'), ax=ax2)
    ax2.set_title('Importance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')

    filtered_spec = get_spectrogram(filtered_audio)
    filtered_db = convert_to_db(filtered_spec)

    # Filtered spectrogram
    fig.colorbar(ax3.imshow(filtered_db[0], aspect='auto', cmap='viridis', origin='lower'), ax=ax3)
    ax3.set_title('Filtered spectogram using importance matrix')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')

    # Normalized importance
    fig.colorbar(ax4.imshow(normalized, aspect='auto', cmap='viridis', origin='lower'), ax=ax4)
    ax4.set_title('Normalized importance')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Frequency')

    plt.show()

    
    # Normalize the spectrogram values to the range [0, 1]
    # norm_spectrogram = (spectogram-np.min(spectogram))/(np.max(spectogram)-np.min(spectogram))

    # Reconstruct the audio signal from the spectrogram
    #audio_original = librosa.istft(spectrogram)

    return original_audio, filtered_audio
    