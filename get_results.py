import argparse
from tools.audio import get_spectrogram, convert_to_db
from tools.relax import prepare_audio, mask_audio, apply_relax, plot_results, extract_masks_features
from prediction import load_beats_model
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm.contrib.itertools import product


# Initialize parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Adding arguments
parser.add_argument('-f', type=str, help='Path to the audio file')

f = parser.parse_args().f
audio_name = f.split('/')[-1].split('.')[0]
folder_name = f.split('/')[-2]
T = [50, 150]
F = [300, 800]
min_t = [0, 50]
min_f = [0, 200]
n_masks = [200, 1000]
n_parts_t = [1, 5]
n_parts_f = [1, 5]
n_ftt = 2048

audios_per_batch = 20

# BEATS model
model_path = '/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/audio/models/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt'
beats_model = load_beats_model(model_path)

audio, sr, cpx_spec, spec_db = prepare_audio(f, plot=True, play=False, resample=True)
# Title on the figure: original spectrogram
fig = plt.imshow(spec_db[0,:,:], origin='lower', cmap='jet', aspect='auto')
fig.axes.set_title('Orignal Spectrogram')

# Joing the paths with os.path.join
home_path = os.path.join('/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/results', folder_name)
plt.savefig(os.path.join(home_path, 'original_spec.png'))

grid = product(T, F, min_t, min_f, n_masks, n_parts_t, n_parts_f)
for params in grid:
    T, F, min_t, min_f, n_masks, n_parts_t, n_parts_f = params
    if min_t >= T or min_f >= F:
        continue
    masked_audio_inputs, all_masks, h_star, masked_audios = mask_audio(audio, sr, cpx_spec, T, F, n_masks, beats_model, min_t, min_f, n_parts_t, n_parts_f)
    # Create and show masked spectrograms
    n_batches = int(n_masks/audios_per_batch)
    h_masks = extract_masks_features(masked_audio_inputs, beats_model, n_batches=n_batches)
    plt.figure(figsize=(15,15))
    for i in range(9):
        # Select random index 
        index = np.random.randint(0, len(masked_audio_inputs))
        tmp_spec = get_spectrogram(masked_audios[index], n_ftt=n_ftt)
        tmp_spec_db = convert_to_db(tmp_spec)
        plt.subplot(3,3,i+1)
        plt.imshow(tmp_spec_db, origin='lower', cmap='jet', aspect='auto')
    # Write the parameters as in the title
    title = 'T: ' + str(T) + ', F: ' + str(F) + ', min_t: ' + str(min_t) + ', min_f: ' + str(min_f) + ', n_masks: ' + str(n_masks) + ', n_parts_t: ' + str(n_parts_t) + ', n_parts_f: ' + str(n_parts_f)
    plt.suptitle(title)
    # Name of the file with the parameters in the format f'{T}_{n_parts}
    file_name = os.path.join(home_path, '{}_{}_{}_{}_{}_{}_{}'.format(T, F, min_t, min_f, n_masks, n_parts_t, n_parts_f))
    plt.savefig(file_name + '.png')
    R, U, W, s = apply_relax(masked_audio_inputs, all_masks, h_masks, h_star, beats_model)
    fig = plot_results(R, U, W, s)
    title = 'T: ' + str(T) + ', F: ' + str(F) + ', min_t: ' + str(min_t) + ', min_f: ' + str(min_f) + ', n_masks: ' + str(n_masks) + ', n_parts_t: ' + str(n_parts_t) + ', n_parts_f: ' + str(n_parts_f)
    fig.subplots_adjust(top=0.90)  # Adjust the top margin
    fig.suptitle(title)
    plt.savefig(os.path.join(home_path, '{}_{}_{}_{}_{}_{}_{}_results.png'.format(T, F, min_t, min_f, n_masks, n_parts_t, n_parts_f)))
