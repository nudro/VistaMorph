#!/bin/bash

# Create output directory
mkdir -p output

# Run inference
python test_VistaMorph_Affine.py \
    --input_dir /data/input \
    --output_dir /data/output \
    --model_path weights/model.pth 