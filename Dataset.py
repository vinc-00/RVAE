import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from collections import defaultdict
from types import SimpleNamespace
from torchvision import datasets, transforms
import torchvision.datasets as tv_datasets
import torchvision.transforms as transforms
import time, os, torch, random
from torch.utils.data import DataLoader, random_split, Dataset
from copy import deepcopy


class TwoDigitRelationMNIST(Dataset):
    def __init__(self, root='./data', min_num=0, max_num=99, relations=[-1, 1, -12, 12]):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist = datasets.MNIST(root=root, train=True, download=True, transform=transform)

        self.data = mnist.data
        self.targets = mnist.targets
        self.relations = []
        self.pairs = []
        self.actual_relations = []
        self.min_num = min_num
        self.max_num = max_num
        relation_to_idx = {-1: 0, 1: 1, -12: 2, 12: 3}

        numbers = list(range(min_num, max_num + 1))
        for num in numbers:
            for rel in relations:
                target_num = num + rel

                if target_num < min_num or target_num > max_num:
                    continue

                num_tens = num // 10
                num_units = num % 10
                target_tens = target_num // 10
                target_units = target_num % 10

                tens_idx = (self.targets == num_tens).nonzero().squeeze()
                units_idx = (self.targets == num_units).nonzero().squeeze()
                target_tens_idx = (self.targets == target_tens).nonzero().squeeze()
                target_units_idx = (self.targets == target_units).nonzero().squeeze()

                if min(len(tens_idx), len(units_idx),
                      len(target_tens_idx), len(target_units_idx)) == 0:
                    continue

                relation_label = relation_to_idx[rel]

                for _ in range(min(350, len(tens_idx))):
                    self.pairs.append((num, target_num))
                    self.relations.append(relation_label)
                    self.actual_relations.append(rel)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_num, tgt_num = self.pairs[idx]
        relation = self.relations[idx]
        actual_rel = self.actual_relations[idx]

        src_tens = src_num // 10
        src_units = src_num % 10
        src_tens_img = self.get_digit_image(src_tens)
        src_units_img = self.get_digit_image(src_units)
        src_img = torch.cat([src_tens_img, src_units_img], dim=-1)  # [1, 28, 56]

        tgt_tens = tgt_num // 10
        tgt_units = tgt_num % 10
        tgt_tens_img = self.get_digit_image(tgt_tens)
        tgt_units_img = self.get_digit_image(tgt_units)
        tgt_img = torch.cat([tgt_tens_img, tgt_units_img], dim=-1)  # [1, 28, 56]

        return src_img.unsqueeze(0), tgt_img.unsqueeze(0), src_num, tgt_num, relation, actual_rel

    def get_digit_image(self, digit):
        indices = (self.targets == digit).nonzero().squeeze()
        idx = indices[torch.randint(0, len(indices), (1,))]
        return self.data[idx].float() / 255.0
