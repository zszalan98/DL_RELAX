#!/bin/bash

# The first argument to this script is the path to the audio file
AUDIO_FILE="$1"

# Submit the job with bsub
bsub -q hpc \
     -J RELAX_$AUDIO_FILE\
     -n 2 \
     -W 24:00 \
     -R "rusage[mem=64GB]" \
     -o /zhome/58/f/181392/DTU/DL/Project/DL_RELAX/outputs/%J.out \
     -e /zhome/58/f/181392/DTU/DL/Project/DL_RELAX/outputs/%J.err \
     "/zhome/58/f/181392/DTU/DL/Project/DL_RELAX/train.sh $AUDIO_FILE"
