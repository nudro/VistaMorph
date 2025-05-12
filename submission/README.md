# VistaMorph Topcoder Submission

## Directory Structure
```
submission/
├── test.py              # Main inference script
├── test.sh             # Test script
├── requirements.txt     # Python dependencies
└── weights/            # Directory for model weights
    ├── net_200.pth
    ├── generator1_200.pth
    └── generator2_200.pth
```

## Weight Files
Place the following weight files in the `weights` directory:
- `net_200.pth`
- `generator1_200.pth`
- `generator2_200.pth`

These files should be the model weights from epoch 200 of training.

## Running the Code
1. Ensure all weight files are in the `weights` directory
2. Run the test script:
   ```bash
   ./test.sh
   ```

The script will:
1. Create input/output directories if they don't exist
2. Run the inference on input .tiff files
3. Generate solution.csv in the output directory 