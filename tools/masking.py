import torch
from torch.nn.functional import interpolate

def create_masks(spec_shape: tuple,
                 n_masks: int = 1000,
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

    tmp_pads_freq = torch.randint(0, pad_freq, (n_masks,))
    tmp_pads_time = torch.randint(0, pad_time, (n_masks,))

    masks = torch.zeros(n_masks, spec_n_freq, spec_n_time).bool()
    for i in range(n_masks):
        slice_freq = slice(tmp_pads_freq[i], tmp_pads_freq[i]+spec_n_freq)
        slice_time = slice(tmp_pads_time[i], tmp_pads_time[i]+spec_n_time)
        masks[i] = masks_with_padding[i, slice_freq, slice_time]

    return masks


def apply_masks(spec: torch.Tensor, masks: torch.Tensor):
    return spec.masked_fill(masks, 0.0)
