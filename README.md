# 3D Reconstruction from Multi-Focus Microscopic Images

This repository builds on the work of Takahiro Yamaguchi [1] 
and Sanchayan Santra [2], which aims to reconstruct 3D models of cells from
multi-focus microscopic images.

### Instructions

To run training on sample, `python main.py -m <model_name> -i <input_imgs_path>`

*Required arguments:*

`-m`: Model selection. Must be in `["dip", "nf", "iter"]`. \
`-i`: Path to folder of same-size 1-channel images.

*Optional arguments:*

`-gt`: Path to slices of 3D GT volume if available. Slices must have same shape as input images. \
`-v`: Version name to track progress in TensorBoard. \
`-p`: Pretrain model. If `"dip"`, must be in `["sc", "v2", "v3"]`. If `"nf"`, must be `"const"`. Weights of network are
saved inside the run's TensorBoard folder. \
`-w`: Path to pretrained model weights. If given along with `-p`, pretraining will be skipped. \
`-n`: Noise level to apply to observed images. Variance of centered normal distribution. Default: 0.

### References

[1] Yamaguchi, T. et al. (2020). 3D Image Reconstruction from Multi-focus Microscopic Images. In: Dabrowski, J., Rahman,
A., Paul, M. (eds) Image and Video Technology. PSIVT 2019. Lecture Notes in Computer Science(), vol 11994. Springer, 
Cham. https://doi.org/10.1007/978-3-030-39770-8_6

[2] https://scholar.google.com/citations?user=ob4KL10AAAAJ&hl=en&authuser=1&oi=ao