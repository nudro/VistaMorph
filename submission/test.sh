#!/bin/bash

# Create necessary directories
mkdir -p input output

# Run the inference script with parameters
python3 test.py \
    --epoch 200 \
    --experiment spacenet_chippedV1FFT \
    --img_height 256 \
    --img_width 256 \
    --channels 3 \
    --input_dir ./input \
    --output_dir ./output \
    --weights_dir ./weights

# Check if solution.csv was created
if [ -f "output/solution.csv" ]; then
    echo "Success: solution.csv was created"
    # Print first few lines of solution.csv
    echo "First few lines of solution.csv:"
    head -n 5 output/solution.csv
else
    echo "Error: solution.csv was not created"
    exit 1
fi 