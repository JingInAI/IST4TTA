import numpy as np
from tqdm import tqdm

import faiss
from faiss import normalize_L2

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import scipy
from scipy.sparse.linalg import cg

from .unlabeled_dataset import PseudoLabelDataset
from .tools import gather_tensor



def soft_labeling(
    samples,
    class_names,
    model_predict,
    preprocess=None,
    extend=1,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    store_loc=False,
    device='cuda',
    return_dst=False,
    verbose=True,
):
    """ Soft labeling
    Args:
        samples (list): List of samples, each sample is PIL image
        class_names (list): List of class names
        model_predict (function): Function to predict class probabilities
        preprocess (function): Preprocess function to be applied to each image
        extend (int): Extend the dataset by {extend} times
        batch_size (int): Batch size
        num_workers (int): Number of workers
        shuffle (bool): Whether to shuffle the dataset
        store_loc (bool): Whether to store the location of each sample
        device (str): Device to use
        return_dst (bool): Whether to return the dataset
        verbose (bool): Whether to show progress bar
    """
    # Create the dataset and dataloader
    dataset = PseudoLabelDataset(samples, class_names, preprocess=preprocess, extend=extend, device=device, verbose=verbose)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=shuffle)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Compute soft labels and features
    with torch.no_grad():
        tbar = tqdm(dataloader, desc="Model Predicting", dynamic_ncols=True) if dist.get_rank() == 0 and verbose else dataloader
        for batch in tbar:
            images, indexes, locations = batch[0].to(device), batch[1].to(device), batch[2]
            soft_labels, features = model_predict(images, class_names)

            indexes = torch.cat(gather_tensor(indexes), dim=0).cpu().numpy()
            soft_labels = torch.cat(gather_tensor(soft_labels), dim=0).cpu().numpy()
            features = torch.cat(gather_tensor(features), dim=0).cpu().numpy()

            if store_loc:
                locations = torch.stack(locations, dim=0).T.contiguous().to(device)
                locations = torch.cat(gather_tensor(locations), dim=0).cpu().numpy()
                dataset.set_locations(indexes, locations)
            else:
                images = torch.cat(gather_tensor(images), dim=0).detach().cpu()
                dataset.set_samples(indexes, images)

            dataset.set_pseudo_labels(indexes, soft_labels)
            dataset.set_features(indexes, features)

    if return_dst:
        return dataset
    else:
        return dataset.get_clean_labels()



def PLCA(
    dataset,
    bank=([], []),
    K=5,
    gamma=3,
    mode='ip',
    repeat=1,
    device='cuda',
    return_dst=False,
    verbose=True,
):
    """ Pseudo-label Correcting Algorithm
    Args:
        dataset (PseudoLabelDataset): Dataset to use
        bank (tuple): Tuple of (features, labels) to use as bank
        K (int): Number of neighbors
        gamma (float): Gamma for computing relations
        mode (str): Mode for computing relations, 'ip' or 'l2'
        repeat (int): Number of times to repeat
        device (str): Device to use
        return_dst (bool): Whether to return the dataset
        verbose (bool): Whether to show progress bar
    """
    # Get soft labels and features
    labels = np.array(bank[1] + dataset.pseudo_labels)
    features = np.array(bank[0] + dataset.features)
    assert None not in labels, "Some samples do not have pseudo labels"
    assert None not in features, "Some samples do not have features"
    assert len(labels) == len(features), "Length of soft labels and features must be the same"

    # Search for neighbors
    X = features.astype('float32')
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    if mode == 'ip':
        index = faiss.GpuIndexFlatIP(res, X.shape[1], flat_config)
        normalize_L2(X)
        K_ = K + 1
    elif mode == 'l2':
        index = faiss.GpuIndexFlatL2(res, X.shape[1], flat_config)
        K_ = K + 2
    else:
        raise ValueError("Mode must be 'ip' or 'l2'")

    index.add(X)
    Sim, Idx = index.search(X, K_)
    Dis = Sim if mode == 'l2' else 1. - Sim

    Dis = torch.tensor(Dis, dtype=torch.float16).to(device)
    Idx = torch.tensor(Idx, dtype=torch.long).to(device)

    # Compute the relations
    relation = 1. - Dis / (Dis.max(dim=-1, keepdims=True)[0] + 1e-8) if mode == 'l2' else 1. - Dis / 2.
    # relation = relation * weight[Idx]
    relation[:, 0] = 1.0
    relation = relation ** gamma
    del Dis, Sim

    # Compute the affinity matrix
    Ref = torch.arange(X.shape[0]).unsqueeze(1).repeat(1, K_).flatten().type(torch.long).to(device)
    Idx = torch.stack([Ref, Idx.flatten()], dim=0)
    A = torch.sparse.HalfTensor(Idx, relation.flatten(0), torch.Size([X.shape[0], X.shape[0]])).to(device)
    del relation, Idx, Ref

    # Compute the Laplacian matrix
    W = A + A.T
    del A

    S = torch.sparse.sum(W, dim=-1).to_dense()
    D = 1. / (torch.sqrt(S + 1e-8) + 1e-8)
    D = D.unsqueeze(0)
    Wn = D * W * D.T
    del W, D, S

    eye_matrix = torch.sparse.HalfTensor(torch.stack([torch.arange(X.shape[0]), torch.arange(X.shape[0])], dim=0), torch.ones(X.shape[0])).to(device)
    Af = (eye_matrix - 0.99 * Wn).cpu()
    del Wn, eye_matrix

    row = Af._indices()[0]
    col = Af._indices()[1]
    data = Af._values()
    shape = Af.size()
    Af = scipy.sparse.csr_matrix((data, (row, col)), shape=shape)
    del row, col, data, shape

    # Propagate the labels
    labels = torch.tensor(labels, dtype=torch.float16).to(device)
    for idx in range(repeat):

        Z = torch.zeros(labels.shape, dtype=torch.float16).to(device)
        tbar = tqdm(range(labels.shape[-1]), desc=f"PLCA {idx+1}/{repeat}", dynamic_ncols=True) if dist.get_rank() == 0 and verbose else range(labels.shape[-1])
        for c in tbar:
            y = labels[:, c]
            y = y / (y.sum(dim=-1) + 1e-8)
            f, _ = cg(Af, y.cpu().numpy(), tol=1e-6, maxiter=20)
            Z[:, c] = torch.tensor(f, dtype=torch.float16).to(device)

        labels = torch.zeros(labels.shape, dtype=torch.float16).to(device)
        labels[range(len(labels)), Z.argmax(dim=-1)] = 1.
    
    labels = labels[len(bank[1]):]
    dataset.set_pseudo_labels(list(range(len(labels))), labels.cpu().numpy())
    
    if return_dst:
        return dataset
    else:
        return dataset.get_clean_labels()



def robust_PLCA(
    samples,
    class_names,
    model_predict,
    preprocess=None,
    extend=1,
    batch_size=32,
    num_workers=4,
    shuffle=False,
    store_loc=False,
    bank=([], []),
    K=5,
    gamma=3,
    mode='ip',
    repeat=1,
    device='cuda',
    return_dst=False,
    verbose=True,
):
    """ Robust pseudo-label clustering algorithm
    Args:
        samples (list): List of samples, each sample is PIL image
        class_names (list): List of class names
        model_predict (function): Function to predict class probabilities
        preprocess (function): Preprocess function to be applied to each image
        extend (int): Extend the dataset by {extend} times
        batch_size (int): Batch size
        num_workers (int): Number of workers
        shuffle (bool): Whether to shuffle the dataset
        store_loc (bool): Whether to store the location of each sample
        bank (tuple): Tuple of feature bank and label bank
        K (int): Number of neighbors
        gamma (float): Gamma
        mode (str): Mode to use, 'ip' or 'l2'
        repeat (int): Number of iterations
        device (str): Device to use
        return_dst (bool): Whether to return the dataset
        verbose (bool): Whether to show progress bar
    """
    # Compute soft labels
    dataset = soft_labeling(
        samples,
        class_names,
        model_predict,
        preprocess=preprocess,
        extend=extend,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        store_loc=store_loc,
        device=device,
        return_dst=True,
        verbose=verbose,
    )
    dataset.soft_labels = np.array(dataset.pseudo_labels)

    # Compute hard labels
    soft_labels = np.array(dataset.pseudo_labels)
    C = soft_labels.shape[-1]

    soft_labels_ = soft_labels.reshape(-1, dataset.extend * C)
    hard_labels = np.argmax(soft_labels_, axis=-1) % C

    dataset.hard_labels = []
    for l in hard_labels:
        hl = np.zeros(C)
        hl[l] = 1.
        dataset.hard_labels.extend([hl] * dataset.extend)

    dataset.hard_labels = np.array(dataset.hard_labels)

    # PLCA
    if repeat > 0 and dist.get_rank() == 0:
        dataset = PLCA(
            dataset=dataset,
            bank=bank,
            K=K,
            gamma=gamma,
            mode=mode,
            repeat=repeat,
            device=device,
            return_dst=True,
            verbose=verbose,
        )
    
    # Broadcast the pseudo labels
    dataset.broadcast_pseudo_labels(src=0)

    if return_dst:
        return dataset
    else:
        return dataset.get_clean_labels()
