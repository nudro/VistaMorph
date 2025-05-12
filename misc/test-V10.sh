#!/bin/bash

# Default values
EPOCH=200
DATASET_NAME="eurecom_warped_pairs"
EXPERIMENT="tfcgan_stn_arar"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epoch)
            EPOCH="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Print configuration with clear separation
echo "=========================================="
echo "Test Configuration:"
echo "------------------------------------------"
echo "Epoch: $EPOCH"
echo "Dataset: $DATASET_NAME"
echo "Experiment: $EXPERIMENT"
echo "=========================================="
echo "Starting test..."
echo "------------------------------------------"

# Run the test script
python test_VistaMorph_V10.py \
    --epoch $EPOCH \
    --dataset_name $DATASET_NAME \
    --experiment $EXPERIMENT

# Check if the test completed successfully
if [ $? -eq 0 ]; then
    echo "------------------------------------------"
    echo "Test completed successfully!"
    echo "Results saved in ./images/test_results/$EXPERIMENT/"
    echo "=========================================="
else
    echo "------------------------------------------"
    echo "Test failed!"
    echo "=========================================="
    exit 1
fi 