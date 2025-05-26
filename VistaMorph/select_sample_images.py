import os
import random
import glob
import shutil

def get_image_pairs(directory):
    # Get all images in directory
    all_images = [f for f in os.listdir(directory) if not f.startswith('.')]
    
    # For Carl dataset
    if 'Carl_Final' in directory:
        # Group by base name (removing extension)
        pairs = {}
        for img in all_images:
            base = img.split('.')[0]
            if base not in pairs:
                pairs[base] = []
            pairs[base].append(img)
    
    # For DEVCOM dataset
    else:
        # Group by everything except the _a_ or _b_ part
        pairs = {}
        for img in all_images:
            # Split at _b_ or _a_
            if '_b_' in img:
                base = img.replace('_b_', '_X_')
            else:
                base = img.replace('_a_', '_X_')
            if base not in pairs:
                pairs[base] = []
            pairs[base].append(img)
    
    return pairs

def select_and_clean(dataset_path, num_pairs=5):
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\nProcessing {split_path}")
        
        # Get all image pairs
        pairs = get_image_pairs(split_path)
        
        # Randomly select pairs
        selected_bases = random.sample(list(pairs.keys()), num_pairs)
        
        # Keep track of files to keep
        files_to_keep = []
        for base in selected_bases:
            files_to_keep.extend(pairs[base])
        
        # Delete all other files
        for file in os.listdir(split_path):
            if file.startswith('.'):
                continue
            if file not in files_to_keep:
                os.remove(os.path.join(split_path, file))
                print(f"Deleted: {file}")
            else:
                print(f"Kept: {file}")

# Process both datasets
datasets = ['data/Carl_Final', 'data/DEVCOM_5perc']
for dataset in datasets:
    print(f"\nProcessing dataset: {dataset}")
    select_and_clean(dataset) 