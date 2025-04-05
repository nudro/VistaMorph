set -ex 
# Generates a newly registered/aligned training set
# Using the 0926_STN_V8 using fBA order
# AwB, AfB, fBA, wBA
python test_vistamorph.py --epoch 200 --dataset_name DEVCOM_5perc --experiment 0926_STN_V8_OG_fBA_Mesh --order fBA
