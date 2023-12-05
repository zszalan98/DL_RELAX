import torch
import matplotlib.pyplot as plt



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