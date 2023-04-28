# TensoIR: Tensorial Inverse Rendering (CVPR 2023)

## [Project Page](https://haian-jin.github.io/TensoIR/) |  [Paper](https://arxiv.org/abs/2304.12461)

This repository contains a pytorch implementation for the paper: [TensoRF: Tensorial Inverse Rendering](https://arxiv.org/abs/2304.12461).

**The code can run well, but it is not well organized. I may re-organize the code when I am available.**

https://user-images.githubusercontent.com/79512936/235218355-0d4177c1-7614-4772-a8ec-44d76a95743f.mp4

#### Tested on Ubuntu 20.04 + Pytorch 1.10.1

Install environment:

```
conda create -n TensoIR python=3.8
conda activate TensoIR
pip install torch==1.10 torchvision
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard loguru plyfile
```

## Dataset

### Downloading

**Please download the dataset and environment maps from the following links and put them in the `./data` folder:**

* [TensoIR-Synthetic](https://zjutvstaff-my.sharepoint.com/personal/1906217881_zjubtv_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F1906217881%5Fzjubtv%5Fcom%2FDocuments%2FZJU%5Flearning%2FResearch%2FTensoIR%2FDataset%2FTensoIR%5FSynthetic&ga=1)
  We provide a TensoIR-Synthetic dataset for training and testing. The dataset is rendered by Blender and consists of four complex synthetic scenes (ficus, lego, armadillo, and hotdog). We use the same camera settings as NeRFactor, so we have 100 training views and 200 test views.
  For each view, we provide the normals map, albedo map and multiple RGB images (11 images) under different lighting conditions.
  **More details about the dataset and our multi-light settings can be found in the supplementary material of our paper.**
* [NeRF-Synthetic](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)
  Original NeRF-Synthetic dataset is not widely used for inverse rendering work, as some scenes of it are not entirely rendered with the environment map and some objects' materials cannot be well handled by the simplfied BRDF model(as discussed in the "limitations" section of our paper's supplementary material). But we still provide the original NeRF-Synthetic dataset to facilitate the analysis of our work.
* [Environment Maps](https://zjutvstaff-my.sharepoint.com/:f:/g/personal/1906217881_zjubtv_com/EmYdVkI-avBDiEcpOmV-N1ABZi9N66yKhj6bPtg4aimU7g?e=4Te39g)
  The file folder has environment maps of different resoluitions($2048 \times 1024$ and $1024 \times 512$). We use the relatively lower resolution environment maps for relighting-testing because of the limited GPU memory, though the G.T. data is rendered by high-resolution environment maps. You can also use the higher resolution environment map for relighting-testing if you have enough GPU memory.

### Generating your own synthetic dataset

We provide the code for generating your own synthetic dataset with your own blender files and Blender software. Please download this [file](https://drive.google.com/file/d/1PNrARJVjamVu_WHC_5gKI91zcqAodTYO/view?usp=sharing) and follow the readme.md file inside it to render your own dataset. The Blender rendering scripts heavily rely on the code provided by [NeRFactor](https://github.com/google/nerfactor). Thanks for its great work!



## Training

### Note:

1. After finishing all training iterations, the training script will automatically render the all test images under the learned lighting condition and save them in the log folder. It will also compute all metrics related to geometry, materials, and novel view synthesis(except for relighting). The results will be saved in the log folder as well.
2. Different scenes have different config files. The main difference of those config files is the different weight value for  `normals_diff_weight`, which controls the how close the predicted normals should be to the derived normals. Larger weight will help preventing the normals prediction from overfiting the surpervised colors, but at the same time it will demage the normals prediction network's ability to predict high-frequency details. **We recommend three values to try: `0.0005`, `0.002`, and `0.005` when you train TensoIR on your own dataset.**


### For pretrained checkpoints and results please see:

[Will be uploaded soon]()


### Training under single lighting condition

```bash
export PYTHONPATH=. && python train_tensoIR.py --config ./configs/single_light/armadillo.txt
```


### Training under rotated multi-lighting conditions

```bash
export PYTHONPATH=. && python train_tensoIR_rotated_multi_lights.py  --config ./configs/multi_light_rotated/hotdog.txt
```


### Training under general multi-lighting conditions

```bash
export PYTHONPATH=. && python train_tensoIR_general_multi_lights.py  --config ./configs/multi_light_general/ficus.txt
```


### (Optional) Training for the original NeRF-Synthetic dataset

We don't do quantitative and qualitative comparisons for the original NeRF-Synthetic dataset in our paper (the reasons have been discussed above), but you can still train TensoIR on the original NeRF-Synthetic dataset for some analysis.

```bash
export PYTHONPATH=. && python train_tensoIR_simple.py --config ./configs/single_light/blender.txt
```

## Testing and Validation


### Rendering with a pre-trained model under learned lighting condition

```bash
export PYTHONPATH=. && python "$training_file" --config "$config_path" --ckpt "$ckpt_path" --render_only 1 --render_test 1
```

`"$training_file"` is the training script you used for training, e.g. `train_tensoIR.py` or `train_tensoIR_rotated_multi_lights.py` or `train_tensoIR_general_multi_lights.py`.

`"$config_path"` is the path to the config file you used for training, e.g. `./configs/single_light/armadillo.txt` or `./configs/multi_light_rotated/hotdog.txt` or `./configs/multi_light_general/ficus.txt`.

`"$ckpt_path"` is the path to the checkpoint you want to test.

The result will be stored in `--basedir` defined in the config file.


### Relighting with a pre-trained model under unseen lighting conditions

```bash
export PYTHONPATH=. && python scripts/relight_importance.py --ckpt "$ckpt_path" --config configs/relighting_test/"$scene".txt --batch_size 800
```

We do light intensity importance sampling for relighting. The sampling results are stored in `--geo_buffer_path` defined in the config file.

`"$ckpt_path"` is the path to the checkpoint you want to test.

`"$scene"` is the name of the scene you want to relight, e.g. `armadillo` or `ficus` or `hotdog` or `lego`.

Reduce the `--batch_size` if you have limited GPU memory.

The line 370 of `scripts/relight_importance.py` specifies the names of environment maps for relighting. You can change it if you want to test other unseen lighting conditions.


### Extracting mesh

The mesh will be stored in the same folder as the checkpoint.

```bash
export PYTHONPATH=. && python scripts/export_mesh.py --ckpt "$ckpt_path" 
```


## Citations

If you find our code or paper helps, please consider citing:

```
@inproceedings{Jin2023TensoIR,
  title={TensoIR: Tensorial Inverse Rendering},
  author={Jin, Haian and Liu, Isabella and Xu, Peijia and Zhang, Xiaoshuai and Han, Songfang and Bi, Sai and Zhou, Xiaowei and Xu, Zexiang and Su, Hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Acknowledgement

The code was built on [TensoRF](https://github.com/apchenstu/TensoRF). Thanks for this great project!
