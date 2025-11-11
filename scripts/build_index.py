# scripts/build_index.py
"""
Script to build semantic search index from video directory
Usage:
    python scripts/build_index.py --video_dir ./videos --model_path ./checkpoints/best.pt --output_dir ./video_index
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys
sys.path.append('src')

from models import VideoAnalysisModel, VideoKeywordModel
from search_indexing import VideoIndexingPipeline, VideoSemanticIndex
from utils import load_labels, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Build video search index")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file")
    parser.add_argument("--output_dir", type=str, default="./video_index", help="Output directory for index")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--batch_process", action="store_true", help="Process videos in batches")
    return parser.parse_args()


def load_model(cfg, model_path, device):
    """Load trained model from checkpoint"""
    # Determine which model to use
    if cfg.get('model', {}).get('use_slowfast', False):
        model = VideoAnalysisModel(
            num_object_classes=cfg['model']['num_object_classes'],
            num_action_classes=cfg['model']['num_action_classes'],
            num_scene_classes=cfg['model']['num_scene_classes'],
            num_keywords=cfg['model']['num_keywords'],
            visual_backbone=cfg['model']['visual_backbone'],
            use_slowfast=True,
            fusion_dim=cfg['model']['fusion_dim']
        )
    else:
        model = VideoKeywordModel(
            vit_name=cfg['model']['visual_backbone'],
            num_labels=cfg['model']['num_keywords'],
            temporal_dim=cfg['model']['temporal_dim'],
            num_temporal_layers=cfg['model'].get('temporal_layers', 6)
        )
    
    # Load checkpoint
    ckpt = torch.load(model_path, map_location=device)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {model_path}")
    return model


def main():
    args = parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    set_seed(cfg.get('misc', {}).get('seed', 42))
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(cfg, args.model_path, device)
    
    # Create indexing pipeline
    print("Initializing indexing pipeline...")
    indexer = VideoIndexingPipeline(
        model=model,
        device=device,
        index_save_dir=args.output_dir
    )
    
    # Index videos
    print(f"Indexing videos from {args.video_dir}")
    video_dir = Path(args.video_dir)
    
    if args.batch_process:
        # Process in batches (for large video collections)
        indexer.index_video_directory(str(video_dir))
    else:
        # Process one by one with progress
        video_files = list(video_dir.glob('*.mp4')) + \
                     list(video_dir.glob('*.avi')) + \
                     list(video_dir.glob('*.mov'))
        
        print(f"Found {len(video_files)} videos")
        
        for i, video_path in enumerate(video_files, 1):
            print(f"\n[{i}/{len(video_files)}] Processing {video_path.name}")
            try:
                video_id = video_path.stem
                metadata = indexer.index_video(str(video_path), video_id)
                print(f"  ✓ Indexed successfully")
                print(f"    Duration: {metadata.duration:.1f}s")
                print(f"    Keywords: {', '.join(metadata.keywords[:5])}")
            except Exception as e:
                print(f"  ✗ Failed: {e}")
    
    # Save index
    print(f"\nSaving index to {args.output_dir}")
    indexer.save_index()
    
    # Print statistics
    stats = indexer.semantic_index.stats()
    print("\n" + "="*50)
    print("Index Statistics:")
    print("="*50)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print("="*50)
    print("\n✓ Indexing complete!")


if __name__ == "__main__":
    main()


# scripts/search_videos.py
"""
Script to search indexed videos
Usage:
    python scripts/search_videos.py --index_dir ./video_index --query "person running"
    python scripts/search_videos.py --index_dir ./video_index --keywords person,running --objects car
"""

import argparse
import sys
sys.path.append('src')

from search_indexing import VideoSemanticIndex, MultimodalQueryEngine
import json


def parse_search_args():
    parser = argparse.ArgumentParser(description="Search indexed videos")
    parser.add_argument("--index_dir", type=str, required=True, help="Index directory")
    parser.add_argument("--query", type=str, help="Text query (requires text encoder)")
    parser.add_argument("--keywords", type=str, help="Comma-separated keywords")
    parser.add_argument("--objects", type=str, help="Comma-separated required objects")
    parser.add_argument("--actions", type=str, help="Comma-separated required actions")
    parser.add_argument("--min_duration", type=float, help="Minimum video duration (seconds)")
    parser.add_argument("--max_duration", type=float, help="Maximum video duration (seconds)")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    return parser.parse_args()


def search_main():
    args = parse_search_args()
    
    # Load index
    print(f"Loading index from {args.index_dir}")
    index = VideoSemanticIndex(embedding_dim=512)
    index.load(args.index_dir)
    
    stats = index.stats()
    print(f"Index contains {stats['num_videos']} videos\n")
    
    # Create query engine
    query_engine = MultimodalQueryEngine(index)
    
    # Parse query parameters
    keywords = args.keywords.split(',') if args.keywords else None
    objects = args.objects.split(',') if args.objects else None
    actions = args.actions.split(',') if args.actions else None
    
    # Execute query
    print("Searching...")
    print(f"  Keywords: {keywords}")
    print(f"  Objects: {objects}")
    print(f"  Actions: {actions}")
    print(f"  Duration: {args.min_duration}-{args.max_duration}s\n")
    
    try:
        results = query_engine.query(
            text_query=args.query,
            required_keywords=keywords,
            required_objects=objects,
            required_actions=actions,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            k=args.k
        )
        
        # Display results
        print(f"Found {len(results)} results:\n")
        print("="*80)
        
        results_list = []
        for i, (metadata, score) in enumerate(results, 1):
            print(f"\n{i}. {metadata.video_id} (score: {score:.3f})")
            print(f"   Path: {metadata.video_path}")
            print(f"   Duration: {metadata.duration:.1f}s @ {metadata.fps:.1f} fps")
            print(f"   Resolution: {metadata.resolution}")
            print(f"   Keywords: {', '.join(metadata.keywords[:10])}")
            print(f"   Objects: {', '.join(metadata.detected_objects[:5])}")
            print(f"   Actions: {', '.join(metadata.detected_actions[:5])}")
            if metadata.transcript:
                print(f"   Transcript: {metadata.transcript[:100]}...")
            
            results_list.append({
                'rank': i,
                'video_id': metadata.video_id,
                'video_path': metadata.video_path,
                'score': score,
                'duration': metadata.duration,
                'keywords': metadata.keywords,
                'objects': metadata.detected_objects,
                'actions': metadata.detected_actions,
                'transcript': metadata.transcript
            })
        
        print("\n" + "="*80)
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'query': {
                        'text': args.query,
                        'keywords': keywords,
                        'objects': objects,
                        'actions': actions,
                        'min_duration': args.min_duration,
                        'max_duration': args.max_duration
                    },
                    'results': results_list
                }, f, indent=2)
            print(f"\nResults saved to {args.output}")
    
    except Exception as e:
        print(f"Search failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    search_main()