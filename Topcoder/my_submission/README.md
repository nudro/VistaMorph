# VistaMorph Submission

This submission implements a feature-guided registration model for satellite imagery alignment. The model uses a combination of feature extraction and spatial transformer networks to predict affine transformation matrices between image pairs.

## Model Architecture

The model consists of:
- Feature extraction network for key point detection
- Spatial transformer network for affine transformation prediction
- Feature matching loss for improved registration accuracy

## Usage

The submission expects:
- Input: .tiff files in the /data/input directory
- Output: Affine transformation matrices (.txt files) in the /data/output directory

Each output file contains a 2x3 affine transformation matrix flattened to a 6-element vector.

## Dependencies

All dependencies are listed in requirements.txt and will be installed automatically in the Docker container.

## Testing

The model can be tested using the provided test.sh script, which will:
1. Load the trained model
2. Process input .tiff files
3. Generate affine transformation matrices for each image patch
4. Save results to the output directory 