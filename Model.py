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
    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, emb_dim=128):
        super().__init__()

        self.relation_embedding = nn.Embedding(5, emb_dim)

        encoder_layer_sizes = [encoder_layer_sizes[0] + emb_dim] + encoder_layer_sizes[1:]

        encoder_layers = []
        for in_size, out_size in zip(encoder_layer_sizes[:-1], encoder_layer_sizes[1:]):
            encoder_layers.append(nn.Linear(in_size, out_size))
            encoder_layers.append(nn.BatchNorm1d(out_size))
            encoder_layers.append(nn.LeakyReLU(0.2))
            encoder_layers.append(nn.Dropout(0.2))

        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_fc = nn.Linear(encoder_layer_sizes[-1], latent_size * 2)

        # Decoder (no relation embedding used here)
        decoder_layer_sizes = [latent_size] + decoder_layer_sizes
        decoder_layers = []
        for i, (in_size, out_size) in enumerate(zip(decoder_layer_sizes[:-1], decoder_layer_sizes[1:])):
            decoder_layers.append(nn.Linear(in_size, out_size))
            if i < len(decoder_layer_sizes) - 2:
                decoder_layers.append(nn.BatchNorm1d(out_size))
                decoder_layers.append(nn.LeakyReLU(0.2))
                decoder_layers.append(nn.Dropout(0.2))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x, relation):
        rel_embed = self.relation_embedding(relation)
        x = torch.cat([x, rel_embed], dim=-1)
        h = self.encoder(x)
        return self.encoder_fc(h)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, relation):
        x = x.view(x.size(0), -1)
        h = self.encode(x, relation)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var, z
