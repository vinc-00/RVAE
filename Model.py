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


class RelationVAE(nn.Module):
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes):
        super().__init__()

        self.relation_embedding = nn.Embedding(4, 16)

        encoder_layers = []
        for i, (in_size, out_size) in enumerate(zip(encoder_layer_sizes[:-1], encoder_layer_sizes[1:])):
            encoder_layers.append(nn.Linear(in_size, out_size))
            encoder_layers.append(nn.BatchNorm1d(out_size))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.2))

        self.encoder_fc = nn.Linear(encoder_layer_sizes[-1], latent_size * 2)
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        input_size = latent_size + 16
        decoder_layer_sizes = [input_size] + decoder_layer_sizes

        for i, (in_size, out_size) in enumerate(zip(decoder_layer_sizes[:-1], decoder_layer_sizes[1:])):
            decoder_layers.append(nn.Linear(in_size, out_size))
            if i < len(decoder_layer_sizes) - 2: 
                decoder_layers.append(nn.BatchNorm1d(out_size))
                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Dropout(0.2))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.encoder_fc(h)

    def decode(self, z, relation):
        rel_embed = self.relation_embedding(relation)
        z = torch.cat([z, rel_embed], dim=-1)
        return torch.sigmoid(self.decoder(z))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, relation):
        x = x.view(x.size(0), -1)  # [batch_size, channels, heigth, width] -> [batch_size, flatten_image]
        h = self.encode(x)
        mu, log_var = torch.chunk(h, 2, dim=-1) # split into mean and variance
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, relation)
        return recon_x, mu, log_var, z