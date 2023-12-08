import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tools.audio import get_spectrogram, convert_to_db, save_audio


def plot_and_save_spec(spec: torch.Tensor, save_folder: Path, og_name: str) -> None:
    """
    Plot a spectrogram.

    Parameters:
        spec (torch.Tensor): Spectrogram of shape (num_features, num_time_steps).
        save_folder (Path): Folder to save the figure.
        og_name (str): Name of the original audio file.

    """
    assert len(spec.shape) == 2, "Spectrogram must be of shape (num_features, num_time_steps)."

    # Original audio's name
    if og_name.endswith('.wav'):
        og_name = og_name[:-4]
    # Save path
    save_path = save_folder.joinpath(f'{og_name}_spec.png')

    # Plot spectrogram
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.colorbar(ax.imshow(spec, aspect='auto', cmap='viridis', origin='lower'), ax=ax)
    ax.set_title('Spectrogram')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    fig.savefig(save_path)


def plot_and_save_masks(masks: torch.Tensor, 
                        save_folder: Path,
                        og_name: str,
                        ) -> torch.Tensor:
    """
    Plot samples from masks.

    Parameters:
        masks (torch.Tensor): Masks of shape (num_masks, num_features, num_time_steps).
        save_folder (Path): Folder to save the figure.
        og_name (str): Name of the original audio file.

    Returns:
        torch.Tensor: Indices of the selected masks.

    """
    assert len(masks.shape) == 3, "Masks must be of shape (num_masks, num_features, num_time_steps)."

    # Original audio's name
    if og_name.endswith('.wav'):
        og_name = og_name[:-4]
    # Save path
    masks_fig_save_path = save_folder.joinpath(f'{og_name}_masks.png')

    # Select 4 random masks
    n_masks = masks.shape[0]
    idx = torch.randint(0, n_masks, (3,))

    # Plot masks as 3x2 grid
    fig, axises = plt.subplots(3, 2, figsize=(12, 12))
    for i, (ax_top, ax_bot) in enumerate(axises):
        # Plot spectrogram
        ax_top.imshow(masks[idx[i]], aspect='auto', cmap='viridis', origin='lower')
        ax_top.set_xlabel('Time')
        ax_top.set_ylabel('Frequency')
        ax_top.set_title(f'Mask {idx[i]+1} spectrogram')
        # Plot histogram
        ax_bot.hist(masks[idx[i]].flatten())
        ax_bot.set_xlabel('Mask value')
        ax_bot.set_ylabel('Frequency')
        ax_bot.set_title(f'Mask {idx[i]+1} histogram')
    fig.suptitle('Mask samples')
    fig.savefig(masks_fig_save_path)

    return idx  # Return the selected indices


def plot_and_save_masked_audio(audio: torch.Tensor,
                               selected_idx: torch.Tensor,
                               save_folder: Path,
                               og_name: str,
                               sample_rate: int = 16000
                               ) -> None:
    """
    Plot samples from masked audio.

    Parameters:
        audio (torch.Tensor): Audio of shape (num_masks, num_features, num_time_steps).
        selected_idx (torch.Tensor): Indices of the selected masks.
        save_folder (Path): Folder to save the figure.
        og_name (str): Name of the original audio file.
        sample_rate (int, optional): Sample rate of the audio. Defaults to 16000.
    
    """

    # Original audio's name
    if og_name.endswith('.wav'):
        og_name = og_name[:-4]

    # Save path
    masked_fig_save_path = save_folder.joinpath(f'{og_name}_masked.png')

    # Get selected audios
    selected_audio = audio[selected_idx]

    # Save selected audios
    for i, a in enumerate(selected_audio):
        masked_audio_save_path = save_folder.joinpath(f'{og_name}_masked_{i+1}.wav')
        save_audio(a, sample_rate, masked_audio_save_path)

    # Get spectrograms
    spec = get_spectrogram(audio)
    spec_db = convert_to_db(spec)

    # Plot masked as 3x2 grid
    fig, axises = plt.subplots(len(selected_idx), 2, figsize=(12, 12))
    for i, (ax_top, ax_bot) in enumerate(axises):
        # Plot spectrogram
        ax_top.imshow(spec_db[selected_idx[i]], aspect='auto', cmap='viridis', origin='lower')
        ax_top.set_xlabel('Time')
        ax_top.set_ylabel('Frequency')
        ax_top.set_title(f'Masked spectrogram {selected_idx[i]+1}')
        # Plot histogram
        ax_bot.hist(spec_db[selected_idx[i]].flatten())
        ax_bot.set_xlabel('Mask value')
        ax_bot.set_ylabel('Frequency')
        ax_bot.set_title(f'Masked histogram {selected_idx[i]+1}')
    fig.suptitle('Masked samples')
    fig.savefig(masked_fig_save_path)


def plot_results(spectrogram: torch.Tensor,
                importance_mx: torch.Tensor, 
                uncertainty_mx: torch.Tensor, 
                similarities: torch.Tensor) -> None:
    """
    Plot the importance and uncertainty matrices.

    Parameters:
        spectrogram (torch.Tensor): Spectrogram of shape (num_features, num_time_steps).
        importance_mx (torch.Tensor): Importance matrix of shape (num_features, num_time_steps).
        uncertainty_mx (torch.Tensor): Uncertainty matrix of shape (num_features, num_time_steps).
        similarities (torch.Tensor): Tensor containing the Manhattan distances.

    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 12))

    # Original spectrogram
    fig.colorbar(ax1.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower'), ax=ax1)
    ax1.set_title('Spectrogram of the original audio')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')

    # Similarity distribution
    ax2.hist(similarities, bins=50)
    ax2.set_xlabel('Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of similarities')

    # Importance
    fig.colorbar(ax3.imshow(importance_mx, aspect='auto', cmap='viridis', origin='lower'), ax=ax3)
    ax3.set_title('Importance')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Frequency')
    
    # Uncertainty
    fig.colorbar(ax4.imshow(uncertainty_mx, aspect='auto', cmap='viridis', origin='lower'), ax=ax4)
    ax4.set_title('Uncertainty')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()
    return fig

def plot_no_uncertainty(spectrogram: torch.Tensor,
                importance_mx: torch.Tensor, 
                similarities: torch.Tensor) -> None:
    """
    Plot the importance and uncertainty matrices.

    Parameters:
        spectrogram (torch.Tensor): Spectrogram of shape (num_features, num_time_steps).
        importance_mx (torch.Tensor): Importance matrix of shape (num_features, num_time_steps).
        uncertainty_mx (torch.Tensor): Uncertainty matrix of shape (num_features, num_time_steps).
        similarities (torch.Tensor): Tensor containing the Manhattan distances.

    """
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(18, 6))

    # Original spectrogram
    fig.colorbar(ax1.imshow(spectrogram, aspect='auto', cmap='viridis', origin='lower'), ax=ax1)
    ax1.set_title('Spectrogram of the original audio')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')

    # Importance
    fig.colorbar(ax2.imshow(importance_mx, aspect='auto', cmap='viridis', origin='lower'), ax=ax2)
    ax2.set_title('Importance')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Frequency')

    # Similarity distribution
    ax3.hist(similarities, bins=50)
    ax3.set_xlabel('Similarity')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Histogram of similarities')


    plt.tight_layout()
    return fig