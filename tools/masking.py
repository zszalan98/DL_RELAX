import torch
import numpy as np
from torch.nn.functional import interpolate


def apply_masks(spec: torch.Tensor, masks: torch.Tensor):
    return spec.masked_fill(masks, 0.0)


def apply_advanced_masks(cpx_spec: torch.Tensor, continuous_masks: torch.Tensor):
    # Check that the continuous masks are in the range [0, 1]
    assert max(continuous_masks) <= 1.0, "Continuous masks must be in the range [0, 1]"
    assert min(continuous_masks) >= 0.0, "Continuous masks must be in the range [0, 1]"
    # Check that the last two dimenensions match between cpx_spec and continuous_masks
    assert cpx_spec.shape[-2:] == continuous_masks.shape[-2:], "The last two dimensions of cpx_spec and continuous_masks must match"

    # Convert to magnitude
    magnitude = torch.abs(cpx_spec)  # Compute the magnitude
    db_scale = 20 * torch.log10(magnitude)  # Convert to decibel scale
    db_scale = torch.clamp(db_scale, min=-80.0)  # Clamp to -80 dB

    # Apply continuous masks
    masked_db_scale = (db_scale + 80.0) * continuous_masks - 80.0
    masked_magnitude = 10 ** (masked_db_scale / 20.0)

    # Mask complex tensor
    multiplier = masked_magnitude / magnitude
    masked_cpx_spec = cpx_spec * multiplier

    return masked_cpx_spec


def create_random_masks(spec_shape: tuple,
                 n_masks: int = 100,
                 n_freq: int = 40, 
                 n_time: int = 25,
                 p: float = 0.5):
    
    spec_n_freq, spec_n_time = spec_shape
    
    # Create random masks from Bernoulli distribution
    low_res_masks = torch.zeros(n_masks, n_freq, n_time).bool()
    low_res_masks.bernoulli_(p)

    # Upscale the masks to the spectrogram size plus padding
    n_freq_with_pad = spec_n_freq * (n_freq+1) // n_freq
    n_time_with_pad = spec_n_time * (n_time+1) // n_time

    with_padding_shape = (n_freq_with_pad, n_time_with_pad)
    masks_with_padding = torch.zeros(n_masks, n_freq_with_pad, n_time_with_pad).bool()
    for i in range(n_masks):
        masks_with_padding[i] = interpolate(low_res_masks[i].unsqueeze(0).unsqueeze(0).float(), size=with_padding_shape).squeeze().bool()

    # Create masks without padding
    pad_freq = n_freq_with_pad - spec_n_freq
    pad_time = n_time_with_pad - spec_n_time

    tmp_pad_start_freq = torch.randint(0, pad_freq, (n_masks,))
    tmp_pad_start_time = torch.randint(0, pad_time, (n_masks,))

    masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    for i in range(n_masks):
        slice_freq = slice(tmp_pad_start_freq[i], tmp_pad_start_freq[i]+spec_n_freq)
        slice_time = slice(tmp_pad_start_time[i], tmp_pad_start_time[i]+spec_n_time)
        masks[i] = masks_with_padding[i, slice_freq, slice_time]

    return masks


def create_time_masks(spec_shape: tuple, 
                      T: int, 
                      min_t: int,
                      n_parts: int,
                      n_masks: int = 1000):
    spec_n_freq, spec_n_time = spec_shape
    masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    total_length = np.random.randint(min_t, T, n_masks) # [min_t, T)
    remaining_length = total_length
    part_length = np.zeros((n_masks, n_parts))
    initial_point = np.zeros((n_masks, n_parts))
    for i in range(n_masks):
        last_point = 0
        for j in range(n_parts):
            if remaining_length[i] == 0:
                break
            if j==n_parts-1:
                part_length[i][j] = remaining_length[i]
            else:
                part_length[i][j] = np.random.randint(0, remaining_length[i]) # [0, remaining_length)
            #print("last_point: "+str(last_point)+"  remaining_length: "+str(remaining_length[i])+"  part_lengt: "+str(part_lengt[i][j]))
            #print("rest of distance", spec_n_time - remaining_length[i])
            initial_point[i][j] = np.random.randint(last_point, spec_n_time - remaining_length[i]) # [0, tau - t)
            last_point = initial_point[i][j] + part_length[i][j]
            remaining_length[i] = remaining_length[i] - part_length[i][j]

    for i in range(n_masks):
        for j in range(n_parts):
            masks[i, :, int(initial_point[i][j]):int(initial_point[i][j] + part_length[i][j])] = True
    return masks

def create_freq_masks(spec_shape: tuple, 
                      F: int,
                      min_f: int,
                      n_parts: int,
                      n_masks: int = 1000):
    spec_n_freq, spec_n_time = spec_shape
    masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    total_length = np.random.randint(min_f, F, n_masks) # [min_f, F)
    remaining_length = total_length
    part_length = np.zeros((n_masks,n_parts))
    initial_point = np.zeros((n_masks,n_parts))
    for i in range(n_masks):
        last_point = 0
        for j in range(n_parts):
            if remaining_length[i] == 0:
                break
            if j==n_parts-1:
                part_length[i][j] = remaining_length[i]
            else:
                part_length[i][j] = np.random.randint(0, remaining_length[i]) # [0, remaining_length)
            initial_point[i][j] = np.random.randint(last_point, spec_n_freq - remaining_length[i]) # [0, tau - f)
            last_point = initial_point[i][j] + part_length[i][j]
            remaining_length[i] = remaining_length[i] - part_length[i][j]

    for i in range(n_masks):
        for j in range(n_parts):
            masks[i, int(initial_point[i][j]):int(initial_point[i][j] + part_length[i][j]), :] = True
    return masks
    


