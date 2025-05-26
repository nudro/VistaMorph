# VistaMorph Submission

This directory contains the Docker solution for the VistaMorph submission.

## Prerequisites

- Docker installed on your system
- Access to a remote server with Docker installed

## Building the Docker Image

1. Navigate to the `code` directory:
   ```bash
   cd VistaMorph/Topcoder/my_submission/code
   ```

2. Build the Docker image:
   ```bash
   docker build -t vistamorph_submission .
   ```

## Running the Docker Container

### Local Testing

1. Run the container with input and output directories mounted:
   ```bash
   docker run -v /path/to/input:/data/input -v /path/to/output:/data/output vistamorph_submission
   ```

### Remote Server Testing

1. Transfer the Docker image to the remote server:
   ```bash
   docker save vistamorph_submission | gzip > vistamorph_submission.tar.gz
   scp vistamorph_submission.tar.gz user@remote_server:/path/to/destination
   ```

2. On the remote server, load the Docker image:
   ```bash
   docker load < vistamorph_submission.tar.gz
   ```

3. Run the container on the remote server:
   ```bash
   docker run -v /path/to/input:/data/input -v /path/to/output:/data/output vistamorph_submission
   ```

## Notes

- Ensure that the input directory contains the necessary data for inference.
- The output directory will contain the results of the inference.
- Adjust the paths as necessary for your specific setup.

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