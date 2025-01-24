from pymongo import MongoClient
from typing import Tuple, List, Dict
from collections import Counter
from tqdm import tqdm
import os
from dotenv import load_dotenv

class HDLDatabaseHandler:
    """Handles all database operations for HDL code analysis."""
    
    def __init__(self):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        self.feature_mapping = {}
        self.pattern_mapping = {}
        
    def load_pattern_data(self) -> Tuple[List[str], List[int]]:
        """Load full-code design pattern data from MongoDB."""
        texts, labels = [], []
        
        cursor = self.collection.find({"analysis": {"$exists": True}})
        unique_patterns = set()
        
        # First pass: collect unique patterns
        for doc in tqdm(cursor, desc="Collecting patterns"):
            pattern = doc.get('analysis', {}).get('design_pattern')
            if pattern:
                unique_patterns.add(str(pattern))
        
        # Create pattern mapping
        self.pattern_mapping = {
            pattern: idx for idx, pattern in enumerate(sorted(unique_patterns))
        }
        
        # Second pass: collect samples
        cursor.rewind()
        for doc in tqdm(cursor, desc="Processing documents"):
            content = doc.get('content')
            pattern = doc.get('analysis', {}).get('design_pattern')
            if content and pattern and pattern in self.pattern_mapping:
                texts.append(content)
                labels.append(self.pattern_mapping[pattern])
        
        print(f"\nCollected {len(texts)} samples with {len(unique_patterns)} unique patterns")
        return texts, labels

    def load_segment_data(self, segmenter) -> Tuple[List[str], List[int], List[str]]:
        """Load pre-labeled segment data using VHDLSegmenter."""
        texts, labels, segment_types = [], [], []
        feature_counts = Counter()

        cursor = self.collection.find({"analysis": {"$exists": True}})
        
        # First pass: count features
        for doc in cursor:
            features = doc.get('analysis', {}).get('key_features', [])
            for feature in features:
                if isinstance(feature, dict) and feature.get('key_feature'):
                    feature_counts[feature['key_feature']] += 1
        
        # Filter common features
        common_features = {feature: count for feature, count 
                         in feature_counts.items() if count >= 10}
        
        self.feature_mapping = {feature: idx for idx, feature 
                              in enumerate(sorted(common_features.keys()))}
        
        # Reset cursor for main collection
        cursor.rewind()
        for doc in tqdm(cursor, desc="Processing documents"):
            content = doc.get('content')
            if not content:
                continue

            segments = segmenter.segment_code(content)
            segment_by_type = {seg.segment_type: seg for seg in segments if seg.segment_type}

            features = doc.get('analysis', {}).get('key_features', [])
            for feature in features:
                if not isinstance(feature, dict):
                    continue
                    
                key_feature = feature.get('key_feature')
                seg_type = feature.get('segment_type')
                
                if key_feature in self.feature_mapping and seg_type in segment_by_type:
                    segment = segment_by_type[seg_type]
                    texts.append(segment.content)
                    labels.append(self.feature_mapping[key_feature])
                    segment_types.append(seg_type)

        return texts, labels, segment_types

    def close(self):
        """Close database connection."""
        self.mongo_client.close()