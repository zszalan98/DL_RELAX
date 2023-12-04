import torch
from tools.audio import read_audio, get_complex_spectrogram, inverse_complex_spectrogram
from tools.masking import create_masks, apply_masks
from tools.audio import resample_audio
from torch.nn.functional import cosine_similarity as cosine_sim
from prediction import load_beats_model

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

def get_unmasked_prediction(audio, sr, model, batch_size):
    res_audio = resample_audio(audio, sr, 16000)
    _, _, h, _  = model.extract_features(res_audio)
    # return  h.expand(batch_size, -1)
    return h

def get_saliency(h_star, model, num_batches, batch_options):
    spec_shape = tuple(batch_options["cplx_spec"].shape[1:3])
    saliency = torch.zeros(spec_shape)
    for _ in range(num_batches):
        raw_masks, x_masked = create_masked_batch(**batch_options)
        _, _, out, _ = model.extract_features(x_masked)
        s = cosine_sim(h_star, out)[:, None, None]
        masked_similarity = raw_masks * s.view(-1, 1, 1)
        saliency += torch.mean(masked_similarity, dim=0)
    return saliency / (num_batches * 0.5)

def get_saliency_var(h_star, saliency, model, num_batches, batch_options):
    spec_shape = tuple(batch_options["cplx_spec"].shape[1:3])
    saliency_var = torch.zeros(spec_shape)   
    for _ in range(num_batches):
        raw_masks, x_masked = create_masked_batch(**batch_options)
        _, _, out, _ = model.extract_features(x_masked)
        s = cosine_sim(h_star, out)[:, None, None]
        var = (s - saliency[None])**2
        masked_var= raw_masks * var
        var = torch.mean(masked_var, dim=0)
        saliency_var += var
    return saliency_var / ((num_batches - 1) * 0.5)


def relax(audio_filename, num_masks, batch_size, model):
    """
    Betched relax algorith based on the RELAX paper's methods computes saliency and saliency var.

    Args:
        audio_filename (string): audio file to load
        num_masks (int): Number of masks to apply.
        bactch_size (int): size of batches.
        model: The loaded model used to extract features.

    Returns:
        R (torch.Tensor): Tensor containing the relaxation values.
        U (torch.Tensor): Tensor containing the uncertainty values.
        W (torch.Tensor): Tensor containing the weight values.
    """
    torch.manual_seed(100)

    #Load audio
    audio, sr = read_audio(audio_filename)
    
    # Create audio spectogram
    n_ftt = 2048
    cplx_spec = get_complex_spectrogram(audio, n_ftt=n_ftt)

    with torch.no_grad():
        num_batches = num_masks // batch_size

        h_star = get_unmasked_prediction(audio, sr, model, batch_size)
        batch_options = {"cplx_spec": cplx_spec, "sr": sr, "batch_size": batch_size, "n_ftt": n_ftt}
        saliency = get_saliency(h_star, model, num_batches, batch_options)
        saliency_var = get_saliency_var(h_star, saliency, model, num_batches, batch_options)
    return saliency, saliency_var

if __name__ == '__main__':
    audio_filename = 'audio/sounds/1-9886-A-49.wav'
    model_path = 'audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
    BEATS_model = load_beats_model(model_path)
    relax(audio_filename=audio_filename, num_masks=100, batch_size=10, model=BEATS_model)