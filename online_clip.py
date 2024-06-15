import os
import argparse
import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import clip
from datasets import load_dataset
from timm.utils import AverageMeter

from utils import set_seed, Logger, set_logger, reduce_tensor
from utils import (
    UnlabeledDatasetV6,
    build_transforms,
    robust_PLCA,
    MomentumUpdate,
    MemoryBank,
    freeze_norm_layer,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ViT-B/32")

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--split", type=str, default="test")

parser.add_argument("--iters", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--max_len", type=int, default=10000, help="Max length of memory bank")

parser.add_argument("--extend", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--pi", type=float, default=0.9)
parser.add_argument("--eval_interval", type=int, default=15)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_model", action="store_true", default=False)
parser.add_argument("--save_path", type=str, default="./results/name/")
parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument("--local-rank", type=int, default=-1, help='node rank for distributed training')
args = parser.parse_args()


# Initialize the environment
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=10800))
dist.barrier()

args.batch_size = args.batch_size // world_size
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = args.seed + dist.get_rank()
set_seed(seed)

args.iter_batch = args.batch_size
if args.batch_size < 32:
    args.batch_size = 32
if args.iter_batch * args.extend < 32:
    args.extend = 32 // args.iter_batch


# Create the logger
experiment_name = f"{args.model}_{args.dataset}_{args.split}".replace("/", "-")
if dist.get_rank() == 0:
    logger = Logger(os.path.join(args.save_path, f"{experiment_name}.log"))
    logger.create_config(args)
    logger.info(args)
    set_logger(logger)


# Load the model
model, preprocess = clip.load(args.model, device=device, jit=False)
model.to(torch.float32)
model_without_ddp = model
model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)


# Create the datasets
adapt_transform = build_transforms(preprocess)

def test_transform(examples):
    if "img" in examples.keys():
        examples["image"] = examples["img"]
    elif "image" in examples.keys():
        examples["img"] = examples["image"]
    examples["image"] = [preprocess(image) for image in examples["image"]]
    if "fine_label" in examples.keys():
        examples["label"] = examples["fine_label"]
    return examples

try:
    train_dataset = load_dataset('./data/datasets/'+args.dataset, split=args.split)
    test_dataset = load_dataset('./data/datasets/'+args.dataset, split=args.split)
except:
    train_dataset = load_dataset(args.dataset, split=args.split)
    test_dataset = load_dataset(args.dataset, split=args.split)

train_dataset.set_transform(test_transform)
test_dataset.set_transform(test_transform)


# Create the dataloaders
def collate_fn(examples):
    batch = {}
    batch["image"] = torch.stack([example["image"] for example in examples])
    batch["label"] = torch.tensor([example["label"] for example in examples])
    batch["img"] = [example["img"] for example in examples]
    return batch

train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.iter_batch, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)


# Create optimizer, scheduler and criterion
param_group = [
    {"params": model_without_ddp.parameters(), "lr": args.lr}]
optimizer = torch.optim.SGD(param_group, weight_decay=1e-3)
momentum = MomentumUpdate(model_without_ddp)

def criterion(outputs, targets):
    loss_i = -targets * torch.log_softmax(outputs, dim=0)
    loss_t = -targets * torch.log_softmax(outputs, dim=1)
    loss = (loss_i + loss_t) / 2.
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)
    return loss


# Compute the pseudo labels
memory_bank = MemoryBank(args.max_len)

def compute_pseudo_labels(dataset, preprocess, model, repeat=args.repeat, verbose=True, extend=args.extend):
    model.eval()

    def clip_predict(images, class_names):
        image_features = model.encode_image(images.to(device))
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names])
        text_features = model.encode_text(text.to(device))

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        soft_labels = (100.0 * image_features_norm @ text_features_norm.T).softmax(dim=-1)

        return soft_labels, image_features
    
    dataset.set_pseudo_labels(
        define_func=robust_PLCA,
        model_predict=clip_predict,
        preprocess=preprocess,
        extend=extend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        store_loc=True,
        bank=memory_bank.get_all(),
        K=50,
        gamma=3,
        mode='l2',
        repeat=repeat,
        device=device,
        return_dst=True,
        verbose=verbose,
    )

    features = dataset.get_features()
    pseudo_labels = dataset.get_pseudo_labels()
    memory_bank.update(features, pseudo_labels)
    if verbose: print(f"Memory Bank Size: {memory_bank.size()}")
    dist.barrier()


# Test the model
max_acc_meter = AverageMeter()
model.eval()
with torch.no_grad():

    if 'label' in test_dataset.features:
        class_names = test_dataset.features['label'].names
    elif 'fine_label' in test_dataset.features:
        class_names = test_dataset.features['fine_label'].names
    else:
        raise NotImplementedError('No label information in the dataset.')

    tbar = tqdm(test_loader, dynamic_ncols=True) if dist.get_rank() == 0 else test_loader
    for batch in tbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        image_features = model_without_ddp.encode_image(images)
        text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names])
        text_features = model_without_ddp.encode_text(text.to(device))

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        soft_labels = (100.0 * image_features_norm @ text_features_norm.T).softmax(dim=-1)

        predictions = torch.argmax(soft_labels, dim=-1)
        acc = torch.sum(predictions == labels) / len(labels)
        acc = reduce_tensor(acc)
        max_acc_meter.update(acc.item(), len(labels) * dist.get_world_size())
        if dist.get_rank() == 0:
            tbar.set_description(f"Target Testing  [ MM-Accuracy {max_acc_meter.val:.4f} ({max_acc_meter.avg:.4f}) ]")

if dist.get_rank() == 0:
    logger.info(f"Target Dataset [ MM-Accuracy {max_acc_meter.avg:.4f} ]")


# Train the model
for batch_idx, batch in enumerate(train_loader):
    loss_meter = AverageMeter()

    temp_dataset = UnlabeledDatasetV6(batch, extend=args.extend, class_names=class_names)
    temp_sampler = torch.utils.data.DistributedSampler(temp_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    temp_loader = DataLoader(temp_dataset, sampler=temp_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if memory_bank.size() < args.max_len:
        compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=0, extend=args.extend)
    else:
        compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=args.repeat, extend=args.extend)

    if memory_bank.size() >= args.max_len:
        model.train()
        freeze_norm_layer(model_without_ddp)
        tbar = tqdm(range(args.iters * len(temp_loader)), dynamic_ncols=True) if dist.get_rank() == 0 else None
        for iter in range(args.iters):
            for images, pseudo_labels, soft_labels in temp_loader:
                optimizer.zero_grad()

                images = images.to(device)
                pseudo_labels = pseudo_labels.to(device)
                soft_labels = soft_labels.to(device)

                image_features = model_without_ddp.encode_image(images)
                text = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names])
                text_features = model_without_ddp.encode_text(text.to(device))

                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                outputs = 100.0 * image_features_norm @ text_features_norm.T

                loss = criterion(outputs, pseudo_labels)
                if args.pi > 0:
                    loss += F.kl_div(torch.log_softmax(outputs, dim=-1), soft_labels, reduction='batchmean')
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), images.size(0))
                if dist.get_rank() == 0:
                    tbar.set_description(f"Target Training [ Batch {batch_idx+1}/{len(train_loader)} | Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) ]")
                    tbar.update(1)
        
        if dist.get_rank() == 0: tbar.close()

        momentum(model_without_ddp, m=args.pi)

        if dist.get_rank() == 0 and args.save_model:
            torch.save(model_without_ddp.state_dict(), os.path.join(args.save_path, f"{experiment_name}_backbone_adapt.pth"))
            print(f"Save model to {args.save_path}")


# Test final model on the target dataset
model.eval()
mm_acc_meter = AverageMeter()

tbar = tqdm(train_loader, desc='Testing', dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
for batch in tbar:
    temp_dataset = UnlabeledDatasetV6(batch, extend=args.extend, class_names=class_names, verbose=False)
    compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=0, verbose=False)

    if dist.get_rank() == 0:
        acc = temp_dataset.acc
        mm_acc_meter.update(acc, len(temp_dataset.samples))

if dist.get_rank() == 0:
    logger.info(f"Target Dataset [ Accuracy {mm_acc_meter.avg:.4f} ]"); print(f"Target Dataset [ Accuracy {mm_acc_meter.avg:.4f} ]")
