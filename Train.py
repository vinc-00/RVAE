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

def train_relation_vae():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    full_dataset = TwoDigitRelationMNIST(min_num=0, max_num=99, relations=[-1, 1, -12, 12])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    vae = RelationVAE(
        encoder_layer_sizes=[1568, 1024, 512, 256],
        latent_size=128,
        decoder_layer_sizes=[256, 512, 1024, 1568]
    ).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4, weight_decay=1e-5)

    best_val_loss = float('inf')
    best_model = None
    epochs_no_improve = 0
    patience = 20
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': []}

    for epoch in range(500):
        epoch_start = time.time()

        # Training phase
        vae.train()
        train_loss = 0
        for batch in train_loader:
            src_imgs = batch[0].to(device)
            tgt_imgs = batch[1].to(device).view(-1, 1568)
            relations = batch[4].to(device)  # Using index 4 for relation labels

            optimizer.zero_grad()
            recon_imgs, mu, logvar, _ = vae(src_imgs, relations)
            loss = loss_fn(recon_imgs, tgt_imgs, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 5)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src_imgs = batch[0].to(device)
                tgt_imgs = batch[1].to(device).view(-1, 1568)
                relations = batch[4].to(device)

                recon_imgs, mu, logvar, _ = vae(src_imgs, relations)
                val_loss += loss_fn(recon_imgs, tgt_imgs, mu, logvar).item()

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)

        print(f"Epoch {epoch+1}: "
              f"Train Loss {train_loss:.4f} | "
              f"Val Loss {val_loss:.4f} | "
              f"Time {epoch_time:.2f}s")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model = deepcopy(vae.state_dict())
            print(f"New best model! Val loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping after {epoch+1} epochs!")
                break

    # Load best model
    if best_model is not None:
        vae.load_state_dict(best_model)

    # Save final model and history
    torch.save({
        'model_state': vae.state_dict(),
        'history': history,
        'best_val_loss': best_val_loss
    }, "relation_vae_model.pt")

    return vae, history