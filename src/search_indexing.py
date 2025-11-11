# src/search_indexing.py
"""
Vector search and indexing for semantic video retrieval
Supports FAISS for efficient similarity search
"""

import numpy as np
import torch
import faiss
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class VideoMetadata:
    """Metadata for indexed video"""
    video_id: str
    video_path: str
    duration: float
    fps: float
    resolution: Tuple[int, int]
    timestamp: str
    
    # Extracted features
    detected_objects: List[str]
    detected_actions: List[str]
    detected_scenes: List[str]
    keywords: List[str]
    transcript: str
    ocr_text: str
    
    # Embedding vector (stored separately in FAISS)
    embedding_id: int


class VideoSemanticIndex:
    """
    Semantic search index using FAISS
    Stores video embeddings and enables similarity search
    """
    
    def __init__(self, embedding_dim=512, index_type='IndexFlatIP'):
        """
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: FAISS index type ('IndexFlatIP', 'IndexIVFFlat', 'IndexHNSW')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        # Initialize FAISS index
        if index_type == 'IndexFlatIP':
            # Inner product (cosine similarity for normalized vectors)
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == 'IndexIVFFlat':
            # For large-scale (millions of videos)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, 100)
            self.needs_training = True
        elif index_type == 'IndexHNSW':
            # Hierarchical navigable small world (fast approximate search)
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Metadata storage (embedding_id -> VideoMetadata)
        self.metadata_store = {}
        self.next_id = 0
        
        # Reverse mapping (video_id -> embedding_id)
        self.video_id_map = {}
    
    def train(self, embeddings: np.ndarray):
        """Train index (required for some index types like IVF)"""
        if hasattr(self, 'needs_training') and self.needs_training:
            print(f"Training index with {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self.needs_training = False
    
    def add_video(
        self,
        embedding: np.ndarray,
        metadata: VideoMetadata
    ) -> int:
        """
        Add a video to the index
        
        Args:
            embedding: Normalized embedding vector (embedding_dim,)
            metadata: VideoMetadata object
        
        Returns:
            embedding_id assigned to this video
        """
        # Ensure embedding is normalized and 2D
        embedding = embedding.reshape(1, -1).astype('float32')
        embedding = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-8)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store metadata
        embedding_id = self.next_id
        metadata.embedding_id = embedding_id
        self.metadata_store[embedding_id] = metadata
        self.video_id_map[metadata.video_id] = embedding_id
        
        self.next_id += 1
        return embedding_id
    
    def add_videos_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: List[VideoMetadata]
    ) -> List[int]:
        """Add multiple videos efficiently"""
        assert len(embeddings) == len(metadata_list)
        
        # Normalize embeddings
        embeddings = embeddings.astype('float32')
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        embedding_ids = []
        for i, metadata in enumerate(metadata_list):
            embedding_id = self.next_id + i
            metadata.embedding_id = embedding_id
            self.metadata_store[embedding_id] = metadata
            self.video_id_map[metadata.video_id] = embedding_id
            embedding_ids.append(embedding_id)
        
        self.next_id += len(embeddings)
        return embedding_ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[VideoMetadata, float]]:
        """
        Search for similar videos
        
        Args:
            query_embedding: Query vector (embedding_dim,)
            k: Number of results
            filter_fn: Optional function to filter results (metadata -> bool)
        
        Returns:
            List of (VideoMetadata, similarity_score) tuples
        """
        # Normalize query
        query = query_embedding.reshape(1, -1).astype('float32')
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Search
        distances, indices = self.index.search(query, k * 2)  # Get more for filtering
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.metadata_store.get(idx)
            if metadata is None:
                continue
            
            # Apply filter if provided
            if filter_fn and not filter_fn(metadata):
                continue
            
            results.append((metadata, float(dist)))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_by_text(
        self,
        text_query: str,
        text_encoder,
        k: int = 10
    ) -> List[Tuple[VideoMetadata, float]]:
        """
        Search videos using text query
        Requires a text encoder (e.g., CLIP text encoder)
        """
        # Encode text to embedding
        with torch.no_grad():
            query_embedding = text_encoder(text_query)
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()
        
        return self.search(query_embedding, k)
    
    def search_by_keywords(
        self,
        keywords: List[str],
        k: int = 10
    ) -> List[VideoMetadata]:
        """
        Search by exact keyword match (inverted index style)
        """
        matching_videos = []
        keywords_lower = [kw.lower() for kw in keywords]
        
        for metadata in self.metadata_store.values():
            # Check if any keyword matches
            video_keywords_lower = [kw.lower() for kw in metadata.keywords]
            if any(kw in video_keywords_lower for kw in keywords_lower):
                matching_videos.append(metadata)
        
        return matching_videos[:k]
    
    def get_by_video_id(self, video_id: str) -> Optional[VideoMetadata]:
        """Retrieve metadata by video ID"""
        embedding_id = self.video_id_map.get(video_id)
        if embedding_id is None:
            return None
        return self.metadata_store.get(embedding_id)
    
    def save(self, save_dir: str):
        """Save index and metadata to disk"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss.index"))
        
        # Save metadata
        with open(save_path / "metadata.pkl", 'wb') as f:
            pickle.dump({
                'metadata_store': self.metadata_store,
                'video_id_map': self.video_id_map,
                'next_id': self.next_id,
                'embedding_dim': self.embedding_dim,
                'index_type': self.index_type
            }, f)
        
        print(f"Index saved to {save_dir}")
    
    def load(self, save_dir: str):
        """Load index and metadata from disk"""
        save_path = Path(save_dir)
        
        # Load FAISS index
        self.index = faiss.read_index(str(save_path / "faiss.index"))
        
        # Load metadata
        with open(save_path / "metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.metadata_store = data['metadata_store']
            self.video_id_map = data['video_id_map']
            self.next_id = data['next_id']
            self.embedding_dim = data['embedding_dim']
            self.index_type = data['index_type']
        
        print(f"Index loaded from {save_dir} ({self.next_id} videos)")
    
    def stats(self) -> Dict:
        """Get index statistics"""
        return {
            'num_videos': self.next_id,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'index_ntotal': self.index.ntotal
        }


class MultimodalQueryEngine:
    """
    Advanced query engine supporting complex multimodal queries
    Examples:
    - "Find videos where a person is running and laughing"
    - "Show me cooking videos with happy music"
    - "Videos containing text 'AI' and showing computers"
    """
    
    def __init__(self, semantic_index: VideoSemanticIndex):
        self.index = semantic_index
    
    def query(
        self,
        text_query: Optional[str] = None,
        visual_query_embedding: Optional[np.ndarray] = None,
        required_objects: Optional[List[str]] = None,
        required_actions: Optional[List[str]] = None,
        required_keywords: Optional[List[str]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        k: int = 10
    ) -> List[Tuple[VideoMetadata, float]]:
        """
        Execute complex multimodal query
        """
        # Define filter function
        def filter_fn(metadata: VideoMetadata) -> bool:
            # Duration filter
            if min_duration and metadata.duration < min_duration:
                return False
            if max_duration and metadata.duration > max_duration:
                return False
            
            # Object filter
            if required_objects:
                objects_lower = [obj.lower() for obj in metadata.detected_objects]
                if not all(obj.lower() in objects_lower for obj in required_objects):
                    return False
            
            # Action filter
            if required_actions:
                actions_lower = [act.lower() for act in metadata.detected_actions]
                if not all(act.lower() in actions_lower for act in required_actions):
                    return False
            
            # Keyword filter
            if required_keywords:
                keywords_lower = [kw.lower() for kw in metadata.keywords]
                if not all(kw.lower() in keywords_lower for kw in required_keywords):
                    return False
            
            return True
        
        # Execute search
        if visual_query_embedding is not None:
            results = self.index.search(visual_query_embedding, k * 2, filter_fn)
        elif text_query:
            # Would need text encoder here
            raise NotImplementedError("Text query requires text encoder")
        else:
            # Just apply filters without similarity search
            results = []
            for metadata in self.index.metadata_store.values():
                if filter_fn(metadata):
                    results.append((metadata, 1.0))
            results = results[:k]
        
        return results


# Example usage and indexing pipeline
class VideoIndexingPipeline:
    """
    Complete pipeline for processing videos and building search index
    """
    
    def __init__(
        self,
        model,
        device='cuda',
        index_save_dir='./video_index'
    ):
        from pipeline import ComprehensiveVideoPipeline
        
        self.model = model
        self.device = device
        self.pipeline = ComprehensiveVideoPipeline()
        self.semantic_index = VideoSemanticIndex(embedding_dim=512)
        self.index_save_dir = index_save_dir
    
    def index_video(self, video_path: str, video_id: str) -> VideoMetadata:
        """Process and index a single video"""
        print(f"Indexing video: {video_id}")
        
        # Process through pipeline
        results = self.pipeline.process_video(video_path)
        model_inputs = self.pipeline.extract_features_for_model(results)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            video = model_inputs['video'].to(self.device)
            audio = model_inputs['audio'].to(self.device) if model_inputs['audio'] is not None else None
            text = model_inputs['text'].to(self.device) if model_inputs['text'] is not None else None
            
            outputs = self.model(video, audio_spec=audio, text_embed=text)
        
        # Extract predictions
        embedding = outputs['semantic_embedding'][0].cpu().numpy()
        
        # Get top predictions for each category
        object_probs = torch.sigmoid(outputs['object_logits'][0]).cpu().numpy()
        action_probs = torch.sigmoid(outputs['action_logits'][0]).cpu().numpy()
        keyword_probs = torch.sigmoid(outputs['keyword_logits'][0]).cpu().numpy()
        
        # Get top-k for each
        top_objects = np.argsort(-object_probs)[:10]
        top_actions = np.argsort(-action_probs)[:10]
        top_keywords = np.argsort(-keyword_probs)[:20]
        
        # Create metadata
        metadata = VideoMetadata(
            video_id=video_id,
            video_path=video_path,
            duration=results['video_metadata']['duration'],
            fps=results['video_metadata']['fps'],
            resolution=results['video_metadata']['resolution'],
            timestamp=datetime.now().isoformat(),
            detected_objects=[f"object_{i}" for i in top_objects],  # Replace with actual labels
            detected_actions=[f"action_{i}" for i in top_actions],
            detected_scenes=[],  # Add scene predictions
            keywords=[f"keyword_{i}" for i in top_keywords],
            transcript=results['audio']['transcript']['text'] if results['audio']['transcript'] else "",
            ocr_text=" ".join([text for _, text in results['ocr']]),
            embedding_id=-1  # Will be set by index
        )
        
        # Add to index
        self.semantic_index.add_video(embedding, metadata)
        
        return metadata
    
    def index_video_directory(self, video_dir: str):
        """Index all videos in a directory"""
        video_paths = list(Path(video_dir).glob('*.mp4')) + \
                     list(Path(video_dir).glob('*.avi')) + \
                     list(Path(video_dir).glob('*.mov'))
        
        print(f"Found {len(video_paths)} videos to index")
        
        for video_path in video_paths:
            video_id = video_path.stem
            try:
                self.index_video(str(video_path), video_id)
            except Exception as e:
                print(f"Failed to index {video_id}: {e}")
        
        # Save index
        self.save_index()
    
    def save_index(self):
        """Save the search index"""
        self.semantic_index.save(self.index_save_dir)
    
    def load_index(self):
        """Load existing search index"""
        self.semantic_index.load(self.index_save_dir)


# Example search interface
def search_videos_example():
    """Example of how to use the search system"""
    
    # Load index
    index = VideoSemanticIndex(embedding_dim=512)
    index.load('./video_index')
    
    # Create query engine
    query_engine = MultimodalQueryEngine(index)
    
    # Example queries
    
    # 1. Search by keywords
    results = query_engine.query(
        required_keywords=['person', 'running'],
        required_objects=['car'],
        k=10
    )
    
    print("Query: Videos with person running near a car")
    for metadata, score in results:
        print(f"  {metadata.video_id}: {score:.3f}")
    
    # 2. Search with duration filter
    results = query_engine.query(
        required_actions=['cooking'],
        min_duration=60.0,
        max_duration=300.0,
        k=5
    )
    
    print("Query: Cooking videos between 1-5 minutes")
    for metadata, score in results:
        print(f"  {metadata.video_id}: {metadata.duration:.1f}s")