# src/train.py
import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from datasets import VideoFrameDataset
from models import EnhancedVideoKeywordModel
from utils import set_seed, load_labels, save_checkpoint
from metrics import precision_at_k_batch, mean_average_precision
import numpy as np
from tqdm import tqdm

def collate_fn(batch):
    vids = [b[0] for b in batch]  # (T, C, H, W)
    labels = [b[1] for b in batch]
    vids = torch.stack(vids)  # (B, T, C, H, W)
    labels = torch.stack(labels)
    return vids, labels

def train_epoch(model, dataloader, optimizer, scaler, device, loss_fn):
    model.train()
    running_loss = 0.0
    for vids, labels in tqdm(dataloader):
        vids = vids.to(device)  # (B, T, C, H, W)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(vids)  # (B, L)
            loss = loss_fn(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * vids.size(0)
    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for vids, labels in tqdm(dataloader):
            vids = vids.to(device)
            logits = model(vids)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    p_at_5 = precision_at_k_batch(all_probs, all_labels, k=5)
    m_ap = mean_average_precision(all_probs, all_labels)
    return p_at_5, m_ap

def main(cfg_path):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get('misc', {}).get('seed', 42))
    labels = load_labels('vocab/labels.json')
    num_labels = len(labels)
    # patch dataset label getter
    VideoFrameDataset.get_num_labels = staticmethod(lambda: num_labels)
    # datasets
    train_ds = VideoFrameDataset(cfg['dataset']['manifest'], num_frames=cfg['dataset']['num_frames'],
                                image_size=cfg['dataset']['image_size'])
    val_ds = VideoFrameDataset(cfg['dataset']['manifest'], num_frames=cfg['dataset']['num_frames'],
                              image_size=cfg['dataset']['image_size'])  # in real life use separate val manifest
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    model = VideoKeywordModel(vit_name=cfg['model']['vit_name'], num_labels=num_labels,
                              temporal_dim=cfg['model']['temporal_dim']).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    scaler = GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    best_map = 0.0
    os.makedirs(cfg['training']['save_path'], exist_ok=True)
    for epoch in range(cfg['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        p_at_5, m_ap = validate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} p@5={p_at_5:.4f} mAP={m_ap:.4f}")
        ckpt_path = os.path.join(cfg['training']['save_path'], f"ckpt_epoch_{epoch}.pt")
        save_checkpoint(model, optimizer, epoch, ckpt_path)
        if m_ap > best_map:
            best_map = m_ap
            save_checkpoint(model, optimizer, epoch, os.path.join(cfg['training']['save_path'], "best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
