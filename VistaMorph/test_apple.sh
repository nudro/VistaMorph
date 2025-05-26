#!/bin/bash
set -ex 
# Generates a newly registered/aligned training set
# Using the Apple M3 version of the test script
# Available orders: AwB, AfB, fBA, wBA
python test_vistamorph_apple.py --epoch 200 --dataset_name DEVCOM_5perc --experiment 0926_STN_V8_OG_fBA_Mesh --order fBA 