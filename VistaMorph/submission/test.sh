#!/bin/bash

# Create necessary directories
mkdir -p input/optical input/sar output

# Check if optical and sar directories exist
if [ ! -d "input/optical" ] || [ ! -d "input/sar" ]; then
    echo "Error: Required directories input/optical and input/sar must exist"
    echo "Please ensure your .tiff files are organized as follows:"
    echo "input/"
    echo "  optical/"
    echo "    image1.tiff"
    echo "    image2.tiff"
    echo "    ..."
    echo "  sar/"
    echo "    image1.tiff"
    echo "    image2.tiff"
    echo "    ..."
    exit 1
fi

# Check if there are matching .tiff files in both directories
optical_files=$(ls input/optical/*.tiff 2>/dev/null | wc -l)
sar_files=$(ls input/sar/*.tiff 2>/dev/null | wc -l)

if [ "$optical_files" -eq 0 ] || [ "$sar_files" -eq 0 ]; then
    echo "Error: No .tiff files found in input/optical or input/sar directories"
    exit 1
fi

if [ "$optical_files" -ne "$sar_files" ]; then
    echo "Error: Number of files in input/optical ($optical_files) does not match input/sar ($sar_files)"
    exit 1
fi

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