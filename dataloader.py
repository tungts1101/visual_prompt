from torchvision import datasets, transforms
from torch.utils import data
import numpy as np
import os
from collections import defaultdict


def get_dataset(config):
    MEAN_CIFAR100 = [0.5071, 0.4867, 0.4408]
    STD_CIFAR100 = [0.2675, 0.2565, 0.2761]
    MEAN_IMAGENET = [0.485, 0.456, 0.406]
    STD_IMAGENET = [0.229, 0.224, 0.225]
    MEAN_CIFAR10 = [0.4914, 0.4822, 0.4465]
    STD_CIFAR10 = [0.2023, 0.1994, 0.2010]
    
    # if config.dataset == 'cifar10':
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
    #     ])
    #     val_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=MEAN_CIFAR10, std=STD_CIFAR10)
    #     ])
    # elif config.dataset == 'cifar100':
    #     train_transform = transforms.Compose([
    #         transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=MEAN_CIFAR100, std=STD_CIFAR100)
    #     ])
    #     val_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=MEAN_CIFAR100, std=STD_CIFAR100)
    #     ])
    # else:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)        
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN_IMAGENET, std=STD_IMAGENET)
    ])
    
    if config.dataset.name == 'cifar10':
        train_dataset = datasets.CIFAR10(config.dataset.root, train=True, download=True, 
                                         transform=train_transform)
        val_dataset = datasets.CIFAR10(config.dataset.root, train=False, download=True, 
                                       transform=val_transform)
        num_classes = 10
    elif config.dataset.name == 'cifar100':
        train_dataset = datasets.CIFAR100(config.dataset.root, train=True, download=True, 
                                          transform=train_transform)
        val_dataset = datasets.CIFAR100(config.dataset.root, train=False, download=True, 
                                        transform=val_transform)
        num_classes = 100
    else:
        raise ValueError(config.dataset.name)
    
    return train_dataset, val_dataset, num_classes


def get_dataloader(config, train_dataset, val_dataset):
    if config.dataset.percentage != 100:
        os.makedirs('dataset', exist_ok=True)
        train_class_indices_filename = f'dataset/{config.dataset.name}_{config.dataset.percentage}_train.npy'
        
        if os.path.exists(train_class_indices_filename):
            all_indices = np.load(train_class_indices_filename, allow_pickle=True).item()
        else:
            all_indices = defaultdict(list)
            for i, (_, label) in enumerate(train_dataset):
                all_indices[label].append(i)
            for label in all_indices.keys():
                np.random.shuffle(all_indices[label])
                all_indices[label] = all_indices[label][:int(len(all_indices[label]) * config.dataset.percentage / 100)]
            np.save(train_class_indices_filename, all_indices)
        
        train_subset_indices = [index for indices in all_indices.values() for index in indices]
        train_subset = data.Subset(train_dataset, train_subset_indices)
    else:
        train_subset = train_dataset
    
    train_loader = data.DataLoader(train_subset, batch_size=config.train.batch_size, 
                                   shuffle=True, num_workers=4)
    val_loader = data.DataLoader(val_dataset, batch_size=config.eval.batch_size, 
                                 shuffle=False, num_workers=4)
    
    return train_loader, val_loader
