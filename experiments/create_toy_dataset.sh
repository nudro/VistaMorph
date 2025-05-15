#!/bin/bash

# Create destination directories
mkdir -p /home/cordun1/vistamorph/data/spacenet_toy/train/labels
mkdir -p /home/cordun1/vistamorph/data/spacenet_toy/test/labels

# Source paths
TRAIN_SRC="/home/cordun1/vistamorph/data/spacenet_chipped2_paired_with_labels/train"
TEST_SRC="/home/cordun1/vistamorph/data/spacenet_chipped2_paired_with_labels/test"

# Destination paths
TRAIN_DST="/home/cordun1/vistamorph/data/spacenet_toy/train"
TEST_DST="/home/cordun1/vistamorph/data/spacenet_toy/test"

# Function to get random files
get_random_files() {
    local src_dir=$1
    local count=$2
    local dst_dir=$3
    
    # Get list of image files (excluding labels)
    local files=($(find "$src_dir" -maxdepth 1 -type f -name "*.png" | shuf -n $count))
    
    # Copy each file and its corresponding label
    for file in "${files[@]}"; do
        # Get filename without path
        filename=$(basename "$file")
        # Get filename without extension
        basename="${filename%.*}"
        
        # Copy image
        cp "$file" "$dst_dir/"
        
        # Copy corresponding label if it exists
        if [ -f "$src_dir/labels/${basename}.txt" ]; then
            cp "$src_dir/labels/${basename}.txt" "$dst_dir/labels/"
        fi
    done
}

# Copy random files for train set
echo "Copying random train files..."
get_random_files "$TRAIN_SRC" 8 "$TRAIN_DST"

# Copy random files for test set
echo "Copying random test files..."
get_random_files "$TEST_SRC" 4 "$TEST_DST"

echo "Toy dataset creation complete!" 