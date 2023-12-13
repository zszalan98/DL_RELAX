import torch
import numpy as np
from torch.nn.functional import interpolate
from tools.audio import convert_to_db, inverse_db


## Apply masks to spectrograms
def apply_masks(spec: torch.Tensor, masks: torch.Tensor):
    # Using convention from the RELAX paper (M_ij = 0 means full mask)
    return spec.masked_fill(not masks, 0.0)


def apply_advanced_masks(cpx_spec: torch.Tensor, continuous_masks: torch.Tensor):
    # Assert type of cpx_spec and continuous_masks
    assert cpx_spec.dtype == torch.complex64, "cpx_spec must be of type torch.complex64"
    assert continuous_masks.dtype == torch.float32, "continuous_masks must be of type torch.float32"
    
    # Check that the last two dimenensions match between cpx_spec and continuous_masks
    assert cpx_spec.shape[-2:] == continuous_masks.shape[-2:], "The last two dimensions of cpx_spec and continuous_masks must match"

    # Convert to magnitude and decibel scale
    db_scale = convert_to_db(cpx_spec) # Clamped to -80 dB
    clamped_magnitude = inverse_db(db_scale)

    # Apply continuous masks
    # Using convention from the RELAX paper (M_ij = 0 means full mask)
    masked_db_scale = (db_scale + 80.0) * continuous_masks - 80.0
    masked_magnitude = inverse_db(masked_db_scale)

    # Mask complex tensor
    multiplier = masked_magnitude / clamped_magnitude
    masked_cpx_spec = cpx_spec * multiplier

    return masked_cpx_spec


## Helper functions for creating masks
def generate_raw_bool_masks(shape: tuple, p: float = 0.5):
    return torch.bernoulli(torch.full(shape, (1-p))).bool()


def generate_raw_masks(shape: tuple, p: float = 0.5):
    return torch.bernoulli(torch.full(shape, (1-p))).float()


def upscale_masks(raw_masks: torch.Tensor, spec_shape: tuple):
    assert len(spec_shape) == 2, "spec_shape must be a tuple of length 2 (freq, time)"
    assert len(raw_masks.shape) == 3, "raw_masks must be a tensor of shape (n_masks, freq, time)"
    # SHAPES
    spec_n_freq, spec_n_time = spec_shape  # Target shape
    n_masks, n_freq, n_time = raw_masks.shape  # Masks shape

    # Upscale the masks to the spectrogram size plus padding
    n_freq_with_pad = spec_n_freq * (n_freq+1) // n_freq
    n_time_with_pad = spec_n_time * (n_time+1) // n_time
    with_padding_shape = (n_freq_with_pad, n_time_with_pad)

    # Upscale the masks with padding using billinear interpolation
    raw_masks = raw_masks.unsqueeze(1).float()  # Add channel dimension and convert to float
    masks_with_padding = interpolate(raw_masks, size=with_padding_shape, mode='bilinear')
    masks_with_padding = masks_with_padding.squeeze(1)  # Remove channel dimension
    
    # Padding sizes
    pad_freq = n_freq_with_pad - spec_n_freq
    pad_time = n_time_with_pad - spec_n_time

    # Crop the masks
    tmp_pad_start_freq = torch.randint(0, pad_freq, (n_masks,))
    tmp_pad_start_time = torch.randint(0, pad_time, (n_masks,))
    masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).float()
    for m in range(n_masks):
        slice_freq = slice(tmp_pad_start_freq[m], tmp_pad_start_freq[m]+spec_n_freq)
        slice_time = slice(tmp_pad_start_time[m], tmp_pad_start_time[m]+spec_n_time)
        masks[m] = masks_with_padding[m, slice_freq, slice_time]

    return masks


## Create masks
def create_relax_masks(spec_shape: tuple,
                       n_masks: int = 100,
                       n_freq: int = 10, 
                       n_time: int = 10,
                       p: float = 0.5,
                       seed: int = None  # For compatibility with MaskingSettings
                       ):
    # Create random masks from Bernoulli distribution
    raw_masks = generate_raw_masks((n_masks, n_freq, n_time), p=p)
    
    # Upscale the masks, interpolate and crop
    masks = upscale_masks(raw_masks, spec_shape[-2:])
    
    return masks


def create_random_masks(spec_shape: tuple,
                 n_masks: int = 100,
                 n_freq: int = 10, 
                 n_time: int = 10,
                 p: float = 0.5,
                 seed: int = None  # For compatibility with MaskingSettings
                 ):
    # Create random masks from Bernoulli distribution
    raw_masks = generate_raw_bool_masks((n_masks, n_freq, n_time), p=p)

    # Upscale the masks, interpolate and crop
    masks = upscale_masks(raw_masks, spec_shape[-2:])

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
    


