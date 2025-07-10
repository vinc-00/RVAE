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
from Util import loss_fn
from Model import RelationVAE
from Dataset import TwoDigitRelationMNIST
from Train import train_relation_vae


model, history = train_relation_vae(v_lr=1e-4, v_epoch=500, v_patience=20, v_model_name='VAE.pt')
model_2, history_2 = train_relation_vae(v_lr=0.00005, v_epoch=500, v_patience=20, v_model_name='VAE.pt')
model_3, history_3 = train_relation_vae(v_lr=0.0005, v_epoch=500, v_patience=20, v_model_name='VAE.pt')