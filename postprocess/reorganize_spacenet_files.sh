#!/bin/bash

# Define the base directory
BASE_DIR="/data/spacenet_chipped/train/data"

# Create the target directories if they don't exist
mkdir -p "/data/spacenet_chipped/train/visible"
mkdir -p "/data/spacenet_chipped/train/sar"

# Loop through each subject directory
for subject_dir in "$BASE_DIR"/subject_*; do
    if [ -d "$subject_dir" ]; then
        # Move visible images
        if [ -d "$subject_dir/visible" ]; then
            mv "$subject_dir/visible"/*.png "/data/spacenet_chipped/train/visible/" 2>/dev/null
        fi
        
        # Move SAR images
        if [ -d "$subject_dir/sar" ]; then
            mv "$subject_dir/sar"/*.png "/data/spacenet_chipped/train/sar/" 2>/dev/null
        fi
    fi
done

echo "File reorganization complete!" 