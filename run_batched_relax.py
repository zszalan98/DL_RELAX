# Import
import torch
from tools.audio import prepare_audio, inverse_complex_spectrogram
from tools.beats import load_beats_model
from tools.masking import create_random_masks, apply_masks
from torch.nn.functional import cosine_similarity as cosine_sim
from tools.batched_relax import update_importance, update_uncertainty, manhattan_similarity
from tools.plotting import plot_results
from pathlib import Path
import matplotlib.pyplot as plt


# Settings classes
class RelaxSettings:
    num_of_batches: int = 360  # Number of batches
    num_of_masks: int = 20  # Number of masks per batch
    
class AudioSettings:
    audio_filename: str = 'rooster_1.wav'  # Audio filename

class MaskingSettings:
    seed: int = 42  # Random seed (needed due to batched processing)
    n_freq: int = 40  # Number of frequency bins
    n_time: int = 25  # Number of time bins
    p: float = 0.6  # Bernoulli distribution parameter


class AllSettings:
    def __init__(self):
        self.relax = RelaxSettings()
        self.audio = AudioSettings()
        self.masking = MaskingSettings()



# Run batched relax function
def run_batched_relax(home_path: Path, settings: AllSettings):
    ## Settings
    num_batches = settings.relax.num_of_batches
    num_masks = settings.relax.num_of_masks

    ## INITIALIZATION
    print('Initializing RELAX...')
    # I. Audio processing
    audio_folder = home_path.joinpath('dataset/selected')
    audio_path = audio_folder.joinpath(settings.audio.audio_filename)

    audio, cpx_spec, spec_db, spec_shape = prepare_audio(audio_path)

    # II. BEATS model
    beats_folder = home_path.joinpath('beats')
    beats_model_path = beats_folder.joinpath('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

    beats_model = load_beats_model(beats_model_path)

    # III. Init RELAX
    importance_mx = torch.zeros(spec_shape)
    uncertainty_mx = torch.zeros(spec_shape)
    mask_weight_mx = torch.ones(spec_shape)
    similarity_mx = torch.zeros((num_batches, num_masks))
    _, _, h_star, test_star = beats_model.extract_features(audio)

    ## RUN IN BATCHES
    print('Running RELAX in batches...')

    for b in range(settings.relax.num_of_batches):
        # Print batch number
        print(f'Batch {b+1}/{settings.relax.num_of_batches}')

        # 1. Create masks
        masks = create_random_masks(spec_shape=spec_shape, n_masks=settings.relax.num_of_masks, **vars(settings.masking))
        mask_weight_mx += torch.mean(masks.float(), dim=0)
        # 2. Apply masks
        masked_cpx_specs = apply_masks(cpx_spec, masks)
        masked_audio_signals = inverse_complex_spectrogram(masked_cpx_specs)
        # 3. Extract features
        _, _, h_masked, test_masked = beats_model.extract_features(masked_audio_signals)
        # 4. Compute similarity
        s = cosine_sim(test_star, test_masked)
        # s = cosine_sim(h_star, h_masked)
        # s = manhattan_similarity(h_star, h_masked)
        similarity_mx[b, :] = s
        # 5. Update RELAX (importance)
        prev_importance_mx = importance_mx.detach()
        importance_mx = update_importance(importance_mx, masks, s)
        uncertainty_mx = update_uncertainty(uncertainty_mx, masks, s, importance_mx, prev_importance_mx)
    final_importance = importance_mx / mask_weight_mx
    final_uncertainty = uncertainty_mx / mask_weight_mx

    # Return results
    return final_importance, final_uncertainty, torch.flatten(s), spec_db[0, :, :]
    

if __name__=="__main__":

    # Path handling
    isWindowsPath = False
    home_path = Path(__file__).parent  # Get parent folder of this file

    # Settings
    settings = AllSettings()
    sound_name = settings.audio.audio_filename.split(".wav")[0]
    if isWindowsPath:
        res_folder =  f"{home_path}\\results\\"
    else:
        res_folder =  f"{home_path}/results/"

    ## MAIN PROGRAM
    torch.manual_seed(settings.masking.seed)
    with torch.no_grad():
        # Run batched relax
        importance, uncertainty, similarities, spec_db = run_batched_relax(home_path, settings)
        torch.save((spec_db, importance, uncertainty, similarities), f"{res_folder}{sound_name}_test2.pt")
        fig = plot_results(spec_db, importance, uncertainty, similarities)
        fig.savefig(f"{res_folder}{sound_name}_test2.png")

