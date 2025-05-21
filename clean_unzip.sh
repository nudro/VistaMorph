#!/bin/bash

# Function to remove Mac hidden files
remove_mac_files() {
    find . -name ".DS_Store" -delete
    find . -name "._*" -delete
    find . -name ".AppleDouble" -delete
    find . -name ".LSOverride" -delete
    find . -name "Icon?" -delete
    find . -name "._*" -delete
    find . -name ".Trashes" -delete
    find . -name "._.Trashes" -delete
    find . -name ".Spotlight-V100" -delete
    find . -name ".fseventsd" -delete
}

# Check if zip file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <zip_file>"
    exit 1
fi

# Unzip the file
unzip "$1"

# Remove Mac hidden files
remove_mac_files

echo "Unzipped and cleaned Mac hidden files from $1" 