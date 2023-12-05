import matplotlib.pyplot as plt 
import numpy as np
import torch
from tqdm import tqdm
from masking import *
from tools.audio import *
from typing import Tuple
from prediction import load_beats_model
from batched_relax import get_saliency, get_saliency_var


def prepare_audio(f, plot=False, play=False, resample=False):
    """
    Prepare audio file for analysis.

    Args:
        f (str): File path of the audio file.
        plot (bool, optional): Whether to plot the spectrogram. Defaults to False.

    Returns:
        tuple: A tuple containing the audio data, sample rate, complex spectrogram, and spectrogram.
    """
    # Read audio
    audio, sr = read_audio(f)

    # Resample audio to 16kHz to save compute time
    if resample:
        audio = resample_audio(audio, sr, 16000)
        sr = 16000

    # Get complex spectrogram
    n_ftt = 2048
    cpx_spec = get_complex_spectrogram(audio, n_ftt=n_ftt)

    # Play audio
    if play:
        play_audio(audio, sr)

    # Plot spectrogram
    if plot:
        spec = get_spectrogram(audio, n_ftt=n_ftt)
        spec_db = convert_to_db(spec)
        # Return the spectrogram for plotting
        return audio, sr, cpx_spec, spec_db
    else:  
        return audio, sr, cpx_spec
    

def mask_spectogram( sr: int, 
                cpx_spec: torch.Tensor, 
                T: int, 
                F: int, 
                n_masks: int, 
                min_t: int = 1, 
                min_f: int = 1, 
                n_parts_t: int = 1,
                n_parts_f: int = 1,
                n_ftt: int = 2048) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(type(n_parts_f))
    """Mask audio and get all masks used

    Args:
        sr (int): sample rate
        cpx_spec (torch.Tensor): _description_
        T (int): max time mask length. 0: no time mask
        F (int): max freq mask length. 0: no freq mask
        n_masks (int): number of masks
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
    res_masked_audios = resample_audio(masked_audios, sr, 16000)
    masked_audio_inputs = torch.reshape(res_masked_audios, (2 * n_masks, 1, res_masked_audios.shape[1]))
    return masked_audio_inputs, all_masks

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


def create_masked_batch(cplx_spec, sr, batch_size, n_ftt):
    # Create spectogram masks
    n_freq, n_time, p = 40, 25, 0.5
    spec_shape = cplx_spec.shape[1:3]
    masks = create_masks(spec_shape, batch_size, n_freq, n_time, p=p)

    # Mask spectogram
    masked_spec = apply_masks(cplx_spec, masks)
    # Retrieve masked audio
    masked_audios = inverse_complex_spectrogram(masked_spec, n_ftt=n_ftt)
    # Resample masked audio
    res_masked_audios = resample_audio(masked_audios, sr, 16000)
    return masks.float(), res_masked_audios

def get_unmasked_prediction(audio, sr, model):
    res_audio = resample_audio(audio, sr, 16000)
    _, _, h, _  = model.extract_features(res_audio)
    # return  h.expand(batch_size, -1)
    return h

def get_saliency(h_star, model, num_batches, batch_options, p=1):
    spec_shape = tuple(batch_options["cplx_spec"].shape[1:3])
    sr = batch_options["sr"]
    cpx_spec = batch_options["spectogram"]
    T = batch_options["T"]
    F = batch_options["F"]
    n_masks = batch_options["batch_size"]

    saliency = torch.zeros(spec_shape)
    for _ in tqdm(range(num_batches), total=num_batches):
        x_masked, raw_masks = mask_spectogram(sr, cpx_spec, T, F, n_masks)
        _, _, out, _ = model.extract_features(x_masked)
        s = torch.cdist(out, h_star, p=p)[:, None, None]
        masked_similarity = raw_masks * s.view(-1, 1, 1)
        saliency += torch.mean(masked_similarity, dim=0)
    return saliency / (num_batches * 0.5)

def get_saliency_var(h_star, saliency, model, num_batches, batch_options, p=1):
    spec_shape = tuple(batch_options["cplx_spec"].shape[1:3])
    sr = batch_options["sr"]
    cpx_spec = batch_options["spectogram"]
    T = batch_options["T"]
    F = batch_options["F"]
    n_masks = batch_options["batch_size"]

    saliency_var = torch.zeros(spec_shape)   
    for _ in tqdm(range(num_batches), total=num_batches):
        x_masked, raw_masks = mask_spectogram(sr, cpx_spec, T, F, n_masks)
        _, _, out, _ = model.extract_features(x_masked)
        s = torch.cdist(out, h_star, p=p)[:, None, None]
        var = (s - saliency[None])**2
        masked_var= raw_masks * var
        var = torch.mean(masked_var, dim=0)
        saliency_var += var
    return saliency_var / ((num_batches - 1) * 0.5)

def apply_batched_relax(n_masks, original_features, model, batch_options):
    """
    Batched relax algorith based on the RELAX paper's methods computes saliency and saliency var.

    Args:
        audio_filename (string): audio file to load
        num_masks (int): Number of masks to apply.
        bactch_size (int): size of batches.
        model: The loaded model used to extract features.

    Returns:
        saliency (torch.Tensor): Tensor containing the importance values.
        saliency_var (torch.Tensor): Tensor containing the uncertainty values.
    """
    n_batches = int(n_masks/batch_options["batch_size"])
    with torch.no_grad():
        saliency = get_saliency(original_features, model, n_batches, batch_options)
        saliency_var = get_saliency_var(original_features, saliency, model, n_batches, batch_options)
    return saliency, saliency_var


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
    f = 'audio/sounds/5-172299-A-5.wav'
    
    # Masking
    T = 200
    F = 200
    n_masks = 100
    p = 1 # 2 for Euclidean distance, 1 for Manhattan distance

    # RELAX
    audios_per_batch = 10
    n_batches = int(n_masks/audios_per_batch)

    # BEATS model
    model_path = 'audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    beats_model = load_beats_model(model_path)

    audio, sr, cpx_spec, spec = prepare_audio(f, plot=True, play=True)
    res_audio = resample_audio(audio, sr, 16000)

    # Extract features of the original audio    
    _, _, h, _  = beats_model.extract_features(res_audio)
    # Expand
    h_star = h.expand(2 * n_masks, -1)


    batch_options = {"spectogram": cpx_spec, "sr": sr, "batch_size": audios_per_batch, "T": T, "F": F}
    R, U, W, s = apply_batched_relax(n_masks, h_star, beats_model, batch_options)
    plot_results(R, U, W, s)
