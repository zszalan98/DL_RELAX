import numpy as np
import matplotlib.pyplot as plt
import librosa
import os

def load_saved_data(spectrogram_file, importance_matrix_file):
    # Load spectrogram and importance matrix from text files
    spectrogram = np.loadtxt(spectrogram_file, delimiter='\t')
    importance_matrix = np.loadtxt(importance_matrix_file, delimiter='\t')

    return spectrogram, importance_matrix

def filter_spectrogram(spectrogram, importance, low_limit):
    # Normalize the spectogram array between 1 and low_limit
    normalized1_0 = (importance-np.min(importance))/(np.max(importance)-np.min(importance))
    normalized = normalized1_0*(1-low_limit)+low_limit

    # Apply filter on the spectogram
    filtered_spectrogram = spectrogram*normalized

    return filtered_spectrogram

def show_results(spectogram, importance, filtered):
    fig, ((ax1, ax2), ax3) = plt.subplots(2, 1, figsize=(12, 12))

    # Original spectrogram
    fig.colorbar(ax1.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower'), ax=ax1)
    ax1.set_title('Spectrogram of the original audio')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')

    # Importance
    fig.colorbar(ax2.imshow(importance, aspect='auto', cmap='viridis', origin='lower'), ax=ax2)
    ax3.set_title('Importance')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')

    # Filtered spectrogram
    fig.colorbar(ax3.imshow(filtered, aspect='auto', cmap='viridis', origin='lower'), ax=ax3)
    ax3.set_title('Filtered spectogram using importance matrix')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')

    # Normalize the spectrogram values to the range [0, 1]
    norm_filterd_spectrogram = (filtered-np.min(filtered))/(np.max(filtered)-np.min(filtered))

    # Reconstruct the audio signal from the spectrogram
    audio_signal = librosa.istft(norm_filterd_spectrogram)

    # Save the reconstructed audio to a WAV file
    librosa.output.write_wav('filtered_sound.wav', audio_signal, sr=22050)
    

if __name__ == "__main__":
    # The directory where the text files are saved
    save_dir = 'experiment'
    low_limit = 0.5

    # Load data from saved text files
    spectrogram_file = os.path.join(save_dir, 'spectrogram.txt')
    importance_matrix_file = os.path.join(save_dir, 'importance_matrix.txt')

    spectrogram, importance_matrix = load_saved_data(spectrogram_file, importance_matrix_file)

    print("Loaded Spectrogram Shape:", spectrogram.shape)
    print("Loaded Importance Matrix Shape:", importance_matrix.shape)

    # Filter spectrogram with the importance_matrix
    filtered = filter_spectrogram(spectrogram, importance_matrix, low_limit)

    # Show the results
    show_results(spectrogram, importance_matrix, filtered)