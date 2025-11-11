# src/infer.py
import argparse
import yaml
import torch
import numpy as np
from models import EnhancedVideoKeywordModel
from datasets import VideoFrameDataset
from utils import load_labels
from decord import VideoReader, cpu
from torchvision import transforms
from PIL import Image
from einops import rearrange

def load_video_frames(path, num_frames=16, image_size=224):
    vr = VideoReader(path, ctx=cpu(0))
    vlen = len(vr)
    if vlen == 0:
        raise ValueError("empty video")
    if vlen <= num_frames:
        indices = list(range(vlen)) + [vlen-1] * (num_frames - vlen)
    else:
        interval = vlen / num_frames
        indices = [int(i * interval) for i in range(num_frames)]
    frames = vr.get_batch(indices).asnumpy()
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    processed = []
    for f in frames:
        pil = Image.fromarray(f)
        t = transform(pil)
        processed.append(t)
    vid = torch.stack(processed)  # T,C,H,W
    return vid

def infer_video(model, device, video_path, labels, cfg):
    vid = load_video_frames(video_path, num_frames=cfg['dataset']['num_frames'],
                            image_size=cfg['dataset']['image_size'])
    vid = vid.unsqueeze(0).to(device)  # 1,T,C,H,W
    model.eval()
    with torch.no_grad():
        logits = model(vid)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    # Get topk
    topk = cfg['inference']['topk']
    idxs = np.argsort(-probs)[:topk]
    threshold = cfg['inference']['threshold']
    results = []
    for i in idxs:
        if probs[i] >= threshold:
            results.append((labels[i], float(probs[i])))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    labels = load_labels("vocab/labels.json")
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    model = VideoKeywordModel(vit_name=cfg['model']['vit_name'], num_labels=len(labels),
                              temporal_dim=cfg['model']['temporal_dim']).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    results = infer_video(model, device, args.video, labels, cfg)
    print("Predicted keywords (label, score):")
    for lab, sc in results:
        print(f"{lab}: {sc:.4f}")

if __name__ == "__main__":
    main()
