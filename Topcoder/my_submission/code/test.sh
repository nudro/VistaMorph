#!/bin/bash

# Create output directory
mkdir -p /data/output

# Run inference
set -ex 
# returns predicted affine matrix
python test_VistaMorph_Affine.py --epoch 150 --input_dir /data/input --output_dir /data/output --weights_dir ./weights --experiment spacenet_chippedV6 