while getopts "f": OPTION

do

python combine_A_and_B_mod.py \
    --experiment ${OPTARG}\
    --fold_A experiments/${OPTARG}/real_A \
    --fold_B experiments/${OPTARG}/real_B \
    --fold_AB experiments/${OPTARG}/pairs/real/ \



done
