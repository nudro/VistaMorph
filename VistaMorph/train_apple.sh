#!/bin/bash
set -x
export PYTORCH_ENABLE_MPS_FALLBACK=1
# data must be paired (Thermal-Visible concatenated image)
# Apple M3 version - uses MPS (Metal Performance Shaders) instead of CUDA
/opt/homebrew/Caskroom/miniconda/base/envs/vistamorph/bin/python vistamorph_apple.py --dataset DEVCOM_5perc --experiment vistamorph1 --batch_size 4 --img_height 256 --img_width 256 --n_epochs 1 