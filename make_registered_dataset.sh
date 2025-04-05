set -ex

# registers the entire training dataset for GAN
# uses the best (official) VTF-STN for Devcom: 0302_STN21_Devcom_NewModel3

python register.py --epoch 50 --experiment your_exp_name --dataset DEVCOM_5perc