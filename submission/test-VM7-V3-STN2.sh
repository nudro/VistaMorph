set -ex 
# the VM7_V3 is the one trained with 2 STNs
python test-VM7-V3-STN2.py --epoch 130 --dataset_name spacenet_toy3 --experiment VM7_V3 
