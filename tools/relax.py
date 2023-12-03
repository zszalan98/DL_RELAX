import matplotlib.pyplot as plt 
import numpy as np
import torch
from tqdm import tqdm
from tools.masking import *
from tools.audio import *
from typing import Tuple
from prediction import load_beats_model


def prepare_audio(f, plot=False, play=False):
    """
    Prepare audio file for analysis.

    Args:
        f (str): File path of the audio file.
        plot (bool, optional): Whether to plot the spectrogram. Defaults to False.

    Returns:
        tuple: A tuple containing the audio data, sample rate, complex spectrogram, and spectrogram.
    """

    audio, sr = read_audio(f)
    n_ftt = 2048
    cpx_spec = get_complex_spectrogram(audio, n_ftt=n_ftt)
    spec = get_spectrogram(audio, n_ftt=n_ftt)
    spec = convert_to_db(spec)
    if plot:
        plt.imshow(spec[0,:,:], origin='lower', cmap='jet', aspect='auto')
    if play:
        play_audio(audio, sr)
    return audio, sr, cpx_spec, spec

def mask_audio(audio: torch.Tensor, 
                sr: int, 
                cpx_spec: torch.Tensor, 
                T: int, 
                F: int, 
                n_masks: int, 
                beats_model, 
                min_t: int = 1, 
                min_f: int = 1, 
                n_parts_t: int = 1,
                n_parts_f: int = 1,
                n_ftt: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(type(n_parts_f))
    """Mask audio and get all masks used

    Args:
        audio (torch.Tensor): audio
        sr (int): sample rate
        cpx_spec (torch.Tensor): _description_
        T (int): max time mask length. 0: no time mask
        F (int): max freq mask length. 0: no freq mask
        n_masks (int): number of masks
        beats_model (Any): beats model
        min_t (int): min time mask length
        min_f (int): min freq mask length
        n_parts_t (int): number of parts to split time mask
        n_parts_f (int): number of parts to split freq mask
        n_ftt (int, optional): Defaults to 2048.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: masked audio, masks, original features
    """
    spec_n_freq, spec_n_time = cpx_spec.shape[1:3] 
    if T>0:
        time_masks = create_time_masks(cpx_spec.shape[1:3], T , min_t, n_parts_t, n_masks)
    else:
        time_masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    if F>0:
        freq_masks = create_freq_masks(cpx_spec.shape[1:3], F , min_f, n_parts_f, n_masks)
    else:
        freq_masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    all_masks = torch.cat((time_masks, freq_masks), dim=0)
    masked_spec_all = apply_masks(cpx_spec, all_masks)
    masked_audios = inverse_complex_spectrogram(masked_spec_all, n_ftt=n_ftt)
    # Resample audio for the BEATS model
    res_audio = resample_audio(audio, sr, 16000)
    res_masked_audios = resample_audio(masked_audios, sr, 16000)

    masked_audio_inputs = torch.reshape(res_masked_audios, (2*n_masks, 1, res_masked_audios.shape[1]))
    # Extract features of the original audio    
    _, _, h, _  = beats_model.extract_features(res_audio)
    h_star = h.expand(2 * n_masks, -1)
    return masked_audio_inputs, all_masks, h_star

def extract_masks_features(masked_audio_inputs, beats_model, n_batches = 100):
    with torch.no_grad():
        # Create empty tensor to store the features
        features = torch.empty((0, 527))
        for i in tqdm(range(n_batches), total=n_batches):
            ix = int(masked_audio_inputs.shape[0]/n_batches)
            x = masked_audio_inputs[ix * i: ix * (i+1), 0, :]
            _, _, temp, _ = beats_model.extract_features(x)
            features = torch.cat((features, temp), dim=0)
    return features


def apply_relax(masked_audio_inputs, masks, h_masks, original_features, beats_model, p=1):
    """
    Applies the RELAX algorithm to compute the relaxation values (R), uncertainty values (U), and weight values (W) 
    for the given masked audio inputs, masks, original features, and beats model.

    Args:
        masked_audio_inputs (torch.Tensor): Tensor containing the masked audio inputs.
        masks (torch.Tensor): Tensor containing the masks.
        original_features (torch.Tensor): Tensor containing the original features.
        beats_model: The beats model used to extract features.

    Returns:
        R (torch.Tensor): Tensor containing the relaxation values.
        U (torch.Tensor): Tensor containing the uncertainty values.
        W (torch.Tensor): Tensor containing the weight values.
    """
    with torch.no_grad():
        W = torch.ones(tuple(masks.shape))
        R = torch.zeros(tuple(masks.shape))
        U = torch.zeros(tuple(masks.shape))
        s = torch.zeros(tuple(masks.shape)[0])
        for mask_idx, x_masked in tqdm(enumerate(masked_audio_inputs), total=masked_audio_inputs.shape[0]):
            raw_mask = masks[mask_idx].float()
            W += raw_mask
            # _, _, h_mask, _ = beats_model.extract_features(x_masked)
            s[mask_idx] = torch.dist(h_masks[mask_idx], original_features, p=p)
            R_prev = R
            R += raw_mask * (s[mask_idx] - R) / W
            U += (s[mask_idx] - R) * (s[mask_idx] - R_prev) * raw_mask
    return R, U, W, s


def plot_results(R: torch.Tensor, U: torch.Tensor, W: torch.Tensor, s: torch.Tensor) -> None:
    """
    Plot the importance and uncertainty matrices.

    Parameters:
        R (torch.Tensor): Importance matrix of shape (num_features, num_time_steps).
        U (torch.Tensor): Uncertainty matrix of shape (num_features, num_time_steps).
        W (torch.Tensor): Weight matrix of shape (num_features, num_time_steps).
        s (torch.Tensor): Tensor containing the Manhattan distances.

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    mean_importance = torch.mean(R, dim=0)
    ax1.imshow(mean_importance, aspect='auto', cmap='viridis', origin='lower')
    ax1.set_title('Importance')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Frequency')
    fig.colorbar(ax1.imshow(mean_importance, aspect='auto', cmap='viridis', origin='lower'), ax=ax1, label='Importance')

    mean_uncertainty = torch.mean(U / (W - 1), dim=0)
    fig.colorbar(ax2.imshow(mean_uncertainty, aspect='auto', cmap='viridis', origin='lower'), ax=ax2, label='Uncertainty')

    ax3.hist(s, bins=100)
    ax3.set_xlabel('Manhattan Distance')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Histogram of Manhattan Distances')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Audio
    f = r'audio/5-172299-A-5.wav'
    
    # Masking
    T = 200
    F = 200
    n_masks = 500
    p = 1 # 2 for Euclidean distance, 1 for Manhattan distance

    # RELAX
    audios_per_batch = 10
    n_batches = int(n_masks/audios_per_batch)

    # BEATS model
    model_path = 'beats_env/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    beats_model = load_beats_model(model_path)

    audio, sr, cpx_spec, spec = prepare_audio(f, plot=True, play=True)
    masked_audio_inputs, all_masks, h_star = mask_audio(audio, sr, cpx_spec, T, F, n_masks, beats_model)
    h_masks = extract_masks_features(masked_audio_inputs, beats_model, n_batches=n_batches)
    R, U, W, s = apply_relax(masked_audio_inputs, all_masks, h_masks, h_star, beats_model)
    plot_results(R, U, W, s)
