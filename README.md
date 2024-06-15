# Improved Self-Training (IST) for Test-Time Adaptation

This respository provides the official implementation of our CVPR 2024 paper:

> Improved Self-Training for Test-Time Adaptation

Please check our [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Improved_Self-Training_for_Test-Time_Adaptation_CVPR_2024_paper.html) for more details.


## Environment

The project is implemented by Python 3.10 and PyTorch 1.13.1.

We adopt CLIP models in our project, so you need to install [CLIP](https://github.com/openai/CLIP) first.

To simplify the dataloading, we employ the "datasets" package from [Hugging Face](https://github.com/huggingface/datasets), which can be installed by

```
pip install datasets
```

Install other dependencies by

```
pip install -r requirements.txt
```

## Datasets

For online test-time adaptation, we consider 7 datasets:

- CIFAR-10

- CIFAR-100

- Food101

- StanfordCars

- ImageNet

- [CIFAR-10C](https://zenodo.org/records/2535967#.Yn5QwbozZhE)

- [ImageNet-C](https://zenodo.org/records/2235448#.Yn5OTrozZhE)

The first five datasets will be automatically downloaded to local.
If it failed due to network problems, please manually download from [Hugging Face](https://huggingface.co/datasets) and store in `./data/datasets/`.
Meanwhile, you need to download CIFAR-10C and ImageNet-C to `./data/CIFAR-10-C` and `./data/ImageNet-C`.

## Run IST

We provide shell scripts in [`scripts/`](scripts/).

Runing IST with CLIP models:

```
bash ./scripts/online_clip.sh
```

Runing IST on CIFAR-10C and ImageNet-C:

```
bash ./scripts/online.sh
```

Runing IST in the setting of continuous TTA:

```
bash ./scripts/continuous_online.sh
```

Please refer to script files for detailed setup.

## Citation

If you find our code useful or our work relevant, please consider citing:

```
@inproceedings{ma2024improved,
  title={Improved Self-Training for Test-Time Adaptation},
  author={Ma, Jing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23701--23710},
  year={2024}
}
```
