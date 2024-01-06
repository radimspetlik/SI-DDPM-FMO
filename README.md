# Evaluation, Training, Demo, and Inference of SI-DDPM-FMO 

### Single-Image Deblurring, Trajectory and Shape Recovery of Fast Moving Objects with Denoising Diffusion Probabilistic Models (WACV 2024)
#### Radim Spetlik, Denys Rozumnyi, Jiri Matas

[[paper](https://openaccess.thecvf.com/content/WACV2024/papers/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.pdf)]
[[cvf](https://openaccess.thecvf.com/content/WACV2024/html/Spetlik_Single-Image_Deblurring_Trajectory_and_Shape_Recovery_of_Fast_Moving_Objects_WACV_2024_paper.html)]

<img src="example/results_siddpmfmo.PNG" width="500">

## Setup

#### Repository clone & environment installation

```
git clone https://github.com/radimspetlik/SI-DDPM-FMO
cd SI-DDPM-FMO
pipenv install
```

Note that we only support pipenv installation. If you do not have pipenv installed, please install it first using ```pip install pipenv```.

## Evaluation

#### 1. Download the pre-trained models

The pre-trained SI-DDPM-FMO models as reported in the paper are available [here](https://drive.google.com/drive/folders/1sS67PAuaKzffSOw6h0pwhKE-Wsvz6nA8?usp=drive_link). 

Download them and place them in the ```models``` dir.

For the baseline method, visit [DeFMO](https://github.com/rozumden/DeFMO), download the pre-trained models and place them it in the ```models``` dir.

#### 2. Download the FMO benchmark dataset

The FMO benchmark datasets are available [here](https://github.com/rozumden/fmo-deblurring-benchmark).
After downloading, place the data in the ```datasets``` dir.

### FMO benchmark evaluation

To evaluate the SI-DDPM-FMO model on the FMO benchmark dataset, run:

```./benchmark_siddmpfmo.sh```

To evaluate the baseline model on the FMO benchmark dataset, run:

```./benchmark_baseline.sh```

## Training

The training scripts will be published soon.

### Synthetic dataset generation
For the dataset generation, please download: 

* [ShapeNetCore.v2 dataset](https://www.shapenet.org/).

* Textures from the [DTD dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/). The exact split used in DeFMO is from the "Neural Voxel Renderer: Learning an Accurate and Controllable Rendering Tool" model and can be downloaded [here](https://polybox.ethz.ch/index.php/s/9Abv3QRm0ZgPzhK).

* Backgrounds for the training dataset from the [VOT dataset](https://www.votchallenge.net/vot2018/dataset.html). 

* Backgrounds for the testing dataset from the [Sports1M dataset](https://cs.stanford.edu/people/karpathy/deepvideo/).

* Blender 2.79b with Python enabled.

Then, insert your paths in renderer/settings.py file. To generate the dataset, run in renderer sub-folder: 
```bash
python run_render.py
```
Note that the full training dataset with 50 object categories, 1000 objects per category, and 24 timestamps takes up to 1 TB of storage memory. Due to this and also the ShapeNet licence, we cannot make the pre-generated dataset public - please generate it by yourself using the steps above.

Reference
------------
If you use this repository, please cite the following [publication](https://arxiv.org/abs/2012.00595):

```bibtex
@inproceedings{siddpmfmo2024,
  author = {Radim Spetlik and Denys Rozumnyi and Jiri Matas},
  title = {Single-Image Deblurring, Trajectory and Shape Recovery of Fast Moving Objects with Denoising Diffusion Probabilistic Models},
  booktitle = {WACV},
  address = {Waikoloa, Hawaii, USA},
  month = jan,
  year = {2024}
}
```
