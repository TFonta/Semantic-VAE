## Automatic Generation of Semantic Parts for Face Image Synthesis
This repository contains the official demo code for the paper:

> **Automatic Generation of Semantic Parts for Face Image Synthesis (ICIAP 2023)** <br>
> Tomaso Fontanini, Claudio Ferrari, Massimo Bertozzi, Andrea Prati <br>
> International Conference on Image Analysis and Processing 2023  

[[ArXiv](https://arxiv.org/abs/2006.03840)]

### Requirements
Code was tested with python 3.9. Other required python packages are pytorch, cv2, PyQt5, einops, PyYAML

### Use the demo
First, download the pre-trained models from this [link](https://univpr-my.sharepoint.com/:u:/g/personal/tomaso_fontanini_unipr_it/Eexk9x0lrNdCgT0OBIu1lMIBHQIlNdRjejEiYNawTo2zwQ?e=RjJdIn).

Create a `checkpoint` folder, place and extract the downloaded zip file into that folder.

Simply run  ```
    bash ./demo.sh
    ```




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