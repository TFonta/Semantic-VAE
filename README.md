## Automatic Generation of Semantic Parts for Face Image Synthesis
This repository contains the official demo code for the paper:

> **Automatic Generation of Semantic Parts for Face Image Synthesis (ICIAP 2023)** <br>
> Tomaso Fontanini, Claudio Ferrari, Massimo Bertozzi, Andrea Prati <br>
> International Conference on Image Analysis and Processing 2023  

Find the paper at the following link: [[ArXiv](https://arxiv.org/pdf/2307.05317.pdf)]

### Requirements
Code was tested with python 3.9. Other required python packages are pytorch, cv2, PyQt5, einops, PyYAML

### Run the code
First, download the pre-trained models from this [Google Drive](https://univpr-my.sharepoint.com/:u:/g/personal/tomaso_fontanini_unipr_it/Eexk9x0lrNdCgT0OBIu1lMIBHQIlNdRjejEiYNawTo2zwQ?e=RjJdIn).

Create a `checkpoint` folder, place and extract the downloaded zip file into that folder.

Simply run  `bash ./demo.sh` with the default parameters. Set `--gpu_ids -1` to run the demo on CPU, or `--gpu_ids 0 [1 .. N]` to run on a GPU with specified GPU_ID.

### How to use the demo
In the `samples` folder, some images and masks from CelebMask-HQ are given.

- Reconstruction: Load an image and the corresponding mask, then click on `edit`
- Style Transfer: Load an image, the corresponding mask and another image, then select a face part and finally on `generate style`
- Generate/Perturb semantic mask parts: Load an image and the corresponding mask and select a face part. Then, click either on `generate` or `perturb` part, and finally on `edit`


### Citation

If you find our work useful, cite us!

```
@inproceedings{fontanini2023automatic,
  title={Automatic Generation of Semantic Parts for Face Image Synthesis},
  author={Fontanini, Tomaso and Ferrari, Claudio and Bertozzi, Massimo and Prati, Andrea},
  booktitle={International Conference on Image Analysis and Processing},
  pages={209--221},
  year={2023},
  organization={Springer}
}
```