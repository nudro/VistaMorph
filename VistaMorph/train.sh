set -ex
# data must be paired (Thermal-Visible concatenated image)
python vistamorph.py --dataset Carl_Final --experiment vistamorph1 --batch_size 32 --gpu_num 0 --n_epochs 100
