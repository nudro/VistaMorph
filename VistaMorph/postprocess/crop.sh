while getopts "f": OPTION

# experiments/images/test_results/0901_STN_V8_OG_fBA

do

  python crop_stn_stack.py --inpath experiments/images/test_results/${OPTARG} \
                              --RA_out experiments/${OPTARG}/real_A \
                                --RB_out experiments/${OPTARG}/real_B \
                                --RegB_out experiments/${OPTARG}/reg_B \
                                --experiment ${OPTARG}
done
