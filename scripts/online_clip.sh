#!/bin/bash
# Task: Online test-time adaptation with CLIP model
# Dataset: CIFAR10, CIFAR100, Food101, ImageNet-1k, StanfordCars
# Model: CLIP

# ViT-B/32
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19345 online_clip.py --model ViT-B/32 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/cifar10_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19346 online_clip.py --model ViT-B/32 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/cifar100_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19347 online_clip.py --model ViT-B/32 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b32/food101_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19372 online_clip.py --model ViT-B/32 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b32/imagenet-1k_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19381 online_clip.py --model ViT-B/32 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b32/stanfordcars_test/

# ViT-B/16
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19348 online_clip.py --model ViT-B/16 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/cifar10_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19349 online_clip.py --model ViT-B/16 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/cifar100_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19350 online_clip.py --model ViT-B/16 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b16/food101_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19373 online_clip.py --model ViT-B/16 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/vit-b16/imagenet-1k_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19382 online_clip.py --model ViT-B/16 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/vit-b16/stanfordcars_test/

# RN50
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19357 online_clip.py --model RN50 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//cifar10_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19358 online_clip.py --model RN50 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//cifar100_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19359 online_clip.py --model RN50 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/rn50//food101_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19376 online_clip.py --model RN50 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/rn50//imagenet-1k_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19385 online_clip.py --model RN50 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/rn50//stanfordcars_test/

# RN101
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19360 online_clip.py --model RN101 --dataset cifar10 --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/cifar10_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19361 online_clip.py --model RN101 --dataset cifar100 --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/cifar100_test/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19362 online_clip.py --model RN101 --dataset food101 --split validation --iters 1 --save_model --save_path ./results/online_clip/rn101/food101_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19377 online_clip.py --model RN101 --dataset imagenet-1k --split validation --iters 1 --save_model --save_path ./results/online_clip/rn101/imagenet-1k_val/
python -m torch.distributed.launch --nproc_per_node 1 --master_port 19386 online_clip.py --model RN101 --dataset Multimodal-Fatima_StanfordCars_test --split test --iters 1 --save_model --save_path ./results/online_clip/rn101/stanfordcars_test/
