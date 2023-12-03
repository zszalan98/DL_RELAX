# clean up outputs from warnings
import warnings
warnings.filterwarnings("ignore")

# Load BEATS model
model_path = 'audio/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
beats_model = load_beats_model(model_path)

# Extract features of the original audio
_, _, h  = beats_model.extract_features(res_audio)
h_star = h.expand(10, -1)
h_star.shape, h.shape


# Batched version
inp_shape = tuple(masks.shape[1:])
mask_bs = 10
num_batches = n_masks // 10

num_batches, inp_shape


torch.manual_seed(100)

with torch.no_grad():
    saliency = torch.zeros(inp_shape)
    saliency_var = torch.zeros(inp_shape)
    split_idx_range = range(mask_bs, res_masked_audios.shape[0], mask_bs)

    for split_idx in split_idx_range:
        x_masked = res_masked_audios[split_idx:split_idx + mask_bs]
        raw_masks = masks[split_idx:split_idx + mask_bs].float()
        _, _, out = beats_model.extract_features(x_masked)
        s = cosine_sim(h_star, out)[:, None, None]
        masked_similarity = raw_masks * s.view(-1, 1, 1)
        saliency += torch.mean(masked_similarity, dim=0)
    saliency /= (num_batches * 0.5)


    for split_idx in split_idx_range:
        x_masked = res_masked_audios[split_idx:split_idx + mask_bs]
        raw_masks = masks[split_idx:split_idx + mask_bs].float()
        _, _, out = beats_model.extract_features(x_masked)
        s = cosine_sim(h_star, out)[:, None, None]
        var = (s - saliency[None])**2
        masked_var= raw_masks * var
        var = torch.mean(masked_var, dim=0)
        saliency_var += var
    saliency_var /= ((num_batches - 1) * 0.5)
saliency, saliency_var