# SWML-DehazeNet: Multi-Scale Image Dehazing via Lightweight Wavelet Domain Learning

## Environment and Dependencies
This implementation requires the following dependencies. We recommend creating a new virtual environment and installing all packages at once.

### 1. Install dependencies via requirements.txt
```bash
# Clone this repository
git clone https://github.com/xiakun-coder/SWML-DehazeNet.git
cd SWML-DehazeNet

# Create and activate a virtual environment (optional but recommended)
conda create -n swml python=3.9.23
conda activate swml

# Install all required packages
pip install -r requirements.txt
```

## Pretrained Weights

Pretrained weights are in `models`

## Test

### Option 1: Testing Images with Quality Metrics
To test hazy images with corresponding ground truth and calculate quantitative metrics (PSNR & SSIM), organize your test dataset as follows:
```
test_input
    |- GT
        |- (image filename)
        |- ...
    |- Hazy
        |- (image filename)
        |- ...
```

Run the testing script: `src/test_IQA.py`

You can select the corresponding pretrained model weights `in src/test_IQA.py`.

The dehazed results will be saved in the `output` folder.A `results.txt` file containing PSNR and SSIM values will also be generated in `output`.

### Option 2: Testing Images without Quality Metrics
To test single hazy images without ground truth (only for visual results), organize your test data as follows:

```
test_input
    |- Hazy
        |- (image filename)
        |- ...
```

Run the testing script:: `src/test.py`

You can select the corresponding pretrained model weights in `src/test.py`.

The dehazed results will be saved in the `output` folder.

### Model Latency and Size
To evaluate model efficiency metrics including latency, FPS, parameter count and memory usage, run: `src/model_latency.py`

The script will generate `latency_results.txt` in `output` containing:
- Model size (parameters and memory footprint)
- Inference speed for different image resolutions
- Frames per second (FPS)

## Contact
If you have any questions or issues contact us via: <xiakun687@gmail.com>