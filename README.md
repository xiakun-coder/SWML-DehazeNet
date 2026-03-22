<<<<<<< HEAD
# WaveLiteDehaze‐Network: A Low‐Parameter Wavelet Based Method for Real‐Time Dehazing

This is the official PyTorch implementation of WaveLiteDehaze-Network (WLD-Net).  


For more details: [[Official Link](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cit2.70011)] 

> **Abstract:** 
While the image dehazing problem has received considerable attention over recent years, the existing models often prioritize performance at the expense of complexity, making them unsuitable for real‐world applications which require algorithms to be deployed on resource constrained‐devices. To address this challenge, we propose WaveLiteDehaze‐Network (WLD‐Net), an end‐to‐end dehazing model that delivers performance comparable to complex models while operating in real‐time and using significantly fewer parameters. This approach capitalizes on the insight that haze predominantly affects low‐frequency information. By exclusively processing the image in the frequency domain using Discrete Wavelet Transform (DWT), we segregate the image into high and low frequencies and process them separately. This allows us to preserve high frequency details and recover low‐frequency components affected by haze, distinguishing our method from existing approaches that use spatial domain processing as the backbone, with DWT serving as an auxiliary component. DWT is applied at multiple levels for better information retention while also accelerating computation by downsampling feature maps. Subsequently, a learning‐based fusion mechanism reintegrates the processed frequencies to reconstruct the dehazed image. Experiments show that WLD‐Net outperforms other low‐parameter models on real‐world hazy images and rivals much larger models, achieving the highest PSNR and SSIM scores on the O‐Haze dataset. Qualitatively, the proposed method demonstrates its effectiveness in handling a diverse range of haze types, delivering visually pleasing results and robust performance, while also generalizing well across different scenarios. With only 0.385 million parameters (more than 100 times smaller than comparable dehazing methods), WLD‐Net processes 1024x1024 images in just 0.045 seconds, highlighting its applicability across various real‐world scenarios.

![WLD-Net](figs/image.jpg)

## Environment and Dependencies:

- CUDA Version: 11.6
- Python: 3.9.12
- torch: 1.12.0


## Pretrained Weights

Pretrained weights are in `models`:

- `DH_dehazing_model_final.pth` - Trained on Dense-Haze dataset
- `NH_dehazing_model_final.pth` - Trained on NH-Haze dataset
- `OH_dehazing_model_final.pth` - Trained on O-Haze dataset
- `RD_dehazing_model_final.pth` - Trained on RESIDE ITS dataset

## Test

### Option 1: Testing Images with Quality Metrics
For testing images with ground truth and generating PSNR & SSIM metrics, organize the test images in the following way:

```
test_input
    |- GT
        |- (image filename)
        |- ...
    |- Hazy
        |- (image filename)
        |- ...
```

Run: `src/test_IQA.py`

In `src/test_IQA.py` you can pick which pretrained model weight to use.

The dehazed images are saved in `output`. It generates a file `results.txt` in `output` with the PSNR and SSIM results.

### Option 2: Testing Images without Quality Metrics
For testing single images without ground truth, organize the test images in the following way:

```
test_input
    |- Hazy
        |- (image filename)
        |- ...
```

Run: `src/test.py`

In `src/test.py` you can pick which pretrained weight to use.

The dehazed images are saved in `output`



### Model Latency and Size
To evaluate model performance metrics (latency, FPS, and model size), run: `src/model_latency.py`

The script will generate `latency_results.txt` in `output` containing:
- Model size (parameters and memory footprint)
- Inference speed for different image resolutions
- Frames per second (FPS)

## Results
![Real World Images](figs/results1.jpg)
![SOTS and O-Haze Images](figs/results2.jpg)

## Contact
If you have any questions or issues contact us via: <ali.murtaza.ali29@outlook.com>

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only.
Any commercial use should get formal permission first.


## 📄 Citation

If you find our repo useful for your research, please cite us:

```bibtex
@article{Murtaza2025,
  title     = {WaveLiteDehaze‐Network: A Low‐Parameter Wavelet‐Based Method for Real‐Time Dehazing},
  author    = {Murtaza,  Ali and Khairuddin,  Uswah and Mohd Faudzi,  Ahmad ’Athif and Hamamoto,  Kazuhiko and Fang,  Yang and Omar,  Zaid},
  journal   = {CAAI Transactions on Intelligence Technology},
  publisher = {Institution of Engineering and Technology (IET)},
  year      = {2025},
  month     = apr,
  ISSN      = {2468-2322},
  doi       = {10.1049/cit2.70011},
  url       = {http://dx.doi.org/10.1049/cit2.70011}
}



=======
# SWML-DehazeNet
SWML-DehazeNet
>>>>>>> a188e511082f28f9e7fa19251eff9d3819f9f249
