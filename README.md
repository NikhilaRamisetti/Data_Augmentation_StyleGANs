# Data_Augmentation_StyleGANs

In one of my repositories I have already trained a vanilla GAN model(please refer to GAN for detail), since 2016 after invention of GAN's lot of its variants were created. But, one that seemed very interseting to me and which was a significant breakthrough was "Style GANs".

```
StyleGAN is a GAN variant that's "better" than traditional GANs by being "style-accurate."
```

StyleGAN is an advanced GAN variant that excels in producing high-quality, customizable, and diverse images. Its ability to disentangle and control specific image attributes and its architectural enhancements make it a powerful tool for various applications, including art, entertainment, fashion, and more.

It is known for:
* High-Quality Image Generation
* Controlled Image Synthesis
* Smooth Interpolation
* No Mode Collapse
* Progressive Growing
* Mapping Networks
* Adaptive Instance Normalization (AdaIN)
* Multi-Resolution Architectures
* State-of-the-Art Image Synthesis
* Transfer Learning

## Getting Started
To get started with this project, follow the code snippets and detailed code is mentioned in the notebook in this repositor
#### **Setting Up the Dataset and Environment**: 

**Dataset:**
https://huggingface.co/datasets/huggan/CelebA-HQ



#### **Pre-trained Model and training further**
In this repository we will be using pre-trained styleGAN model of nvidia.

[StyleGAN3 pre-trained models](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/stylegan3) for config T (translation equiv.) and config R (translation and rotation equiv.)
```
Access individual networks via https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/<MODEL>, where <MODEL> is one of:
stylegan3-t-ffhq-1024x1024.pkl, stylegan3-t-ffhqu-1024x1024.pkl, stylegan3-t-ffhqu-256x256.pkl
stylegan3-r-ffhq-1024x1024.pkl, stylegan3-r-ffhqu-1024x1024.pkl, stylegan3-r-ffhqu-256x256.pkl
stylegan3-t-metfaces-1024x1024.pkl, stylegan3-t-metfacesu-1024x1024.pkl
stylegan3-r-metfaces-1024x1024.pkl, stylegan3-r-metfacesu-1024x1024.pkl
stylegan3-t-afhqv2-512x512.pkl
stylegan3-r-afhqv2-512x512.pkl
stylegan-celebahq-1024x1024.pkl
```
We are using stylegan-celebahq-1024x1024.pkl pre-trained model on celebA hq dataset
```python3
!python train.py --outdir=./results --cfg=stylegan3-t --data=/content/drive/MyDrive/stylegan_xl/data/unsplash-landscapes-1024.zip \
--gpus=1 --batch=32 --batch-gpu=4 --gamma=10.0 --mirror=1 --kimg=5000 --snap=1 \
--resume=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan-celebahq-1024x1024.pkl
```

### Generating images on trained dataset
Here are the results for the same celebA dataset

```python3
!python gen_images.py --help
```

```python3
```
