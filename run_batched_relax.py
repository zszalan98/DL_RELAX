# Import
import torch
from tools.audio import prepare_audio, inverse_complex_spectrogram
from tools.beats import load_beats_model
from tools.masking import create_random_masks, apply_masks, create_relax_masks, apply_advanced_masks
from torch.nn.functional import cosine_similarity as cosine_sim
from tools.batched_relax import update_importance, update_uncertainty, manhattan_similarity
from tools.convergence import get_batch_conv_info
from tools.plotting import plot_results, plot_and_save_masks, plot_and_save_spec, plot_and_save_masked_audio
from tools.experiment_filter_sound import filter_complex_spectrogram, get_reconstructed_audios
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt


# Settings classes
class RelaxSettings:
    def __init__(self):
        self.num_of_batches = 10  # Number of batches
        self.num_of_masks = 10  # Number of masks per batch

class AudioSettings:
    def __init__(self):
        self.audio_filename = 'rooster_1.wav'  # Audio filename

class MaskingSettings:
    def __init__(self):
        self.n_freq = 10  # Number of frequency bins
        self.n_time = 10  # Number of time bins
        self.p = 0.35  # Bernoulli distribution parameter
        self.seed = 42  # Random seed (needed due to batched processing)


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
    # Debug files
    debug_folder = home_path.joinpath('results/debug')
    og_name = settings.audio.audio_filename

    ## INITIALIZATION
    print('Initializing RELAX...')
    # I. Audio processing
    audio_folder = home_path.joinpath('dataset/selected')
    audio_path = audio_folder.joinpath(settings.audio.audio_filename)

    audio, cpx_spec, spec_db, spec_shape = prepare_audio(audio_path)

    plot_and_save_spec(spec_db[0, :, :], debug_folder, og_name)

    # II. BEATS model
    beats_folder = home_path.joinpath('beats')
    beats_model_path = beats_folder.joinpath('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
    beats_model = load_beats_model(beats_model_path)

    # III. Init RELAX
    importance_mx = torch.zeros(spec_shape)
    uncertainty_mx = torch.zeros(spec_shape)
    mask_weight_mx = torch.ones(spec_shape)
    similarity_mx = torch.zeros((num_batches, num_masks))
    # Clamping the original audio
    init_mask = torch.ones(spec_shape)
    init_cpx_spec = apply_advanced_masks(cpx_spec, init_mask)
    init_audio = inverse_complex_spectrogram(init_cpx_spec)
    _, _, h_star, test_star = beats_model.extract_features(init_audio)

    # Init convergence diagnostics
    spec_info = get_batch_conv_info(spec_db)
    batch_info_shape = (num_batches, spec_info.shape[1], spec_info.shape[2], spec_info.shape[3])
    b_info_imp = torch.zeros(batch_info_shape)
    b_info_unc = torch.zeros(batch_info_shape)

    ## RUN IN BATCHES
    print('Running RELAX in batches...')

    for b in range(num_batches):
        # Print batch number
        print(f'Batch {b+1}/{num_batches}')

        # 1. Create masks
        masks = create_random_masks(spec_shape=spec_shape, n_masks=num_masks, **vars(settings.masking))
        mask_weight_mx += torch.mean(masks.float(), dim=0)      
        # 2. Apply masks
        masked_cpx_specs = apply_advanced_masks(cpx_spec, masks)
        masked_audio_signals = inverse_complex_spectrogram(masked_cpx_specs)
        # +++ Debug files
        if b == 0:
            selected_idx = plot_and_save_masks(masks, debug_folder, og_name)
            plot_and_save_masked_audio(masked_audio_signals, selected_idx, debug_folder, og_name)
        # 3. Extract features
        _, _, h_masked, test_masked = beats_model.extract_features(masked_audio_signals)
        # 4. Compute similarity
        s = cosine_sim(test_star, test_masked)
        # s = cosine_sim(h_star, h_masked)
        # s = manhattan_similarity(h_star, h_masked)
        similarity_mx[b, :] = s
        # 5. Update RELAX (importance)
        prev_importance_mx = importance_mx.detach()
        importance_mx = update_importance(importance_mx, masks, s, b)
        uncertainty_mx = update_uncertainty(uncertainty_mx, masks, s, importance_mx, prev_importance_mx, b)
        # +++ Convergence diagnostics
        b_info_imp[b, :, :, :] = get_batch_conv_info(importance_mx)
        b_info_unc[b, :, :, :] = get_batch_conv_info(uncertainty_mx)

    # Get final importance and uncertainty
    final_imp = importance_mx
    final_unc = uncertainty_mx

    # Return results
    return spec_db[0, :, :], final_imp, final_unc, torch.flatten(similarity_mx), b_info_imp, b_info_unc, cpx_spec
    

if __name__=="__main__":

    # Path handling
    isWindowsPath = True
    home_path = Path(__file__).parent  # Get parent folder of this file

    # Settings
    settings = AllSettings()
    sound_name = settings.audio.audio_filename.split(".wav")[0]

    ## MAIN PROGRAM
    torch.manual_seed(settings.masking.seed)
    with torch.no_grad():
        # Run batched relax
        spec_db, importance, uncertainty, similarities, b_imp, b_unc, cpx_spec = run_batched_relax(home_path, settings)

        # Plot results
        fig = plot_results(spec_db, importance, uncertainty, similarities)

        # Obtain filtered audio
        # Low limit for the filter. The lower it is the more agressive the filter
        low_limit = 0.1 # Can go from  1 to 0
        filtered, filtered_norm = filter_complex_spectrogram(cpx_spec, importance, low_limit)
        original_audio, filtered_audio = get_reconstructed_audios(spec_db, importance, filtered, filtered_norm, cpx_spec)


        # Save results
        save_folder = home_path.joinpath('results')
        fig_name_str = f"{sound_name}_test.png"
        tensor_name_str = f"{sound_name}_test.pt"
        save_audio = Path('experiment')

        torchaudio.save(save_audio.joinpath(f'{sound_name}_non_filtered_sound.wav'), original_audio, sample_rate=16000)
        torchaudio.save(save_audio.joinpath(f'{sound_name}_filtered_sound.wav'), filtered_audio, sample_rate=16000)

        fig.savefig(save_folder.joinpath(fig_name_str))
        torch.save((spec_db, importance, uncertainty, similarities, b_imp, b_unc), save_folder.joinpath(tensor_name_str))



