## Post-Processing

The `results` directory contains several scripts for post-processing the model outputs:

### Cropping and Image Organization
- `crop_stn_stack.py`: Processes the test phase output images (256x768 stacked images) by cropping them into individual components:
  - Real visible image (real_A)
  - Real thermal image (real_B)
  - Registered thermal image (reg_B)
  Usage: `./crop.sh -f <experiment_name>`

### Face Mesh Analysis
- `google_face_mesh.py`: Implements the MediaPipe Face Mesh detection to draw facial landmarks on images
  - Uses MediaPipe's face mesh model with 468 landmarks
  - Includes iris detection (refine_landmarks=True)
  - Draws tesselation, contours, and iris connections

- `mesh.py`: Batch processes images to generate face mesh visualizations
  - Runs after cropping is complete
  - Creates separate directories for mesh visualizations:
    - real_A_mesh: Face mesh on visible images
    - real_B_mesh: Face mesh on thermal images
    - reg_B_mesh: Face mesh on registered thermal images
  Usage: `python mesh.py --experiment <experiment_name>`

### Image Pairing Scripts
- `pair.sh`: Creates paired images of real visible (A) and real thermal (B) images
  - Output: `experiments/<experiment>/pairs/real/`
  Usage: `./pair.sh -f <experiment_name>`

- `pair_reg.sh`: Creates paired images of real visible (A) and registered thermal (B) images
  - Output: `experiments/<experiment>/pairs/reg/`
  Usage: `./pair_reg.sh -f <experiment_name>`

### Directory Structure After Post-Processing
```
experiments/<experiment_name>/
├── real_A/           # Cropped visible images
├── real_B/           # Cropped thermal images
├── reg_B/           # Cropped registered thermal images
├── real_A_mesh/     # Face mesh visualizations for visible images
├── real_B_mesh/     # Face mesh visualizations for thermal images
├── reg_B_mesh/      # Face mesh visualizations for registered images
└── pairs/
    ├── real/        # Paired real visible-thermal images
    └── reg/         # Paired visible-registered thermal images
```

Note: All scripts use relative paths with `experiments/` as the base directory. Make sure to run the scripts in sequence: first cropping, then mesh generation (if needed), and finally pairing. 