import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    AutoConfig
)
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
import wandb
import json
import traceback
from torch import nn
# Import custom modules
from utils.HDL_dataset import HDLDataset
from utils.model_logger import ModelLogger
from multi_label import EnhancedMultilabelTrainer, create_enhanced_model

######################################################################
######################################################################
######################################################################

class HDLModelTrainer:
    """Handles training and evaluation of HDL analysis models using labeled data."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        # Check for HF token and initialize wandb
        if not os.getenv('HF_TOKEN'):
            raise ValueError("HF_TOKEN environment variable must be set for HuggingFace Hub access")
        
        if os.getenv('WANDB_API_KEY'):
            wandb.init(project="hdl-analysis", name=f"training-{datetime.now().strftime('%Y%m%d_%H%M')}")
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = ModelLogger().logger
        
        self.label_mappings = {
            'design_pattern': {},   # Will be populated from data
            'key_features': {}      # Will be populated from data
        }
    def load_training_data(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """Load labeled HDL code with minimal filtering."""
        codes, labels = [], {'design_pattern': [], 'key_features': []}
        design_patterns, key_features = set(), set()
        
        try:
            cursor = self.collection.find({"analysis": {"$exists": True}, "content": {"$exists": True}})
            self._extract_patterns_and_features(cursor, design_patterns, key_features)
            self._map_labels(design_patterns, key_features)
            self._gather_data_and_labels(cursor, codes, labels)
            self._save_label_mappings()
            return codes, labels
            
        except Exception as e:
            self._handle_load_data_error(e)

######################################################################
######################################################################
######################################################################

    def _extract_patterns_and_features(self, cursor, design_patterns, key_features):
        """Collect unique design patterns and key features."""
        for doc in cursor:
            analysis = doc.get('analysis', {})
            pattern = analysis.get('design_pattern')
            features = analysis.get('key_features', [])
            if pattern: design_patterns.add(str(pattern))
            key_features.update(str(feature) for feature in features if feature)

######################################################################
######################################################################
######################################################################
    def _map_labels(self, design_patterns, key_features):
        """Create mappings for design patterns and key features."""
        self.label_mappings['design_pattern'] = {pattern: idx for idx, pattern in enumerate(sorted(design_patterns))}
        self.label_mappings['key_features'] = {feature: idx for idx, feature in enumerate(sorted(key_features))}

######################################################################
######################################################################
######################################################################

    def _gather_data_and_labels(self, cursor, codes, labels):
        """Collect HDL code and corresponding labels."""
        cursor.rewind()
        for doc in cursor:
            content = doc.get('content')
            if not content: continue
            pattern = str(doc.get('analysis', {}).get('design_pattern', ''))
            features = doc.get('analysis', {}).get('key_features', [])
            
            codes.append(content)
            labels['design_pattern'].append(self.label_mappings['design_pattern'].get(pattern, 0))
            
            # Ensure the feature vector has a fixed length
            feature_vector = [0] * len(self.label_mappings['key_features'])  # Initialize with zeros
            for feature in features:
                if str(feature) in self.label_mappings['key_features']:
                    feature_vector[self.label_mappings['key_features'][str(feature)]] = 1
            labels['key_features'].append(feature_vector)

######################################################################
######################################################################
######################################################################

    def _save_label_mappings(self):
        """Save label mappings to a JSON file."""
        with open('label_mappings.json', 'w') as f:
            json.dump(self.label_mappings, f, indent=2)

######################################################################
######################################################################
######################################################################

    def _handle_load_data_error(self, e):
        """Handle errors during data loading."""
        print(f"Error in load_training_data: {str(e)}")
        traceback.print_exc()
        raise

######################################################################
######################################################################
######################################################################

    def prepare_datasets(self) -> Dict[str, Dict[str, HDLDataset]]:
        """Prepare training and validation datasets focusing on design patterns and key features."""
        try:
            codes, labels = self.load_training_data()
            if not codes:
                raise ValueError("No training data available")
                
            datasets = {}
            
            for label_type in ['design_pattern', 'key_features']:
                if not labels[label_type]:
                    raise ValueError(f"No labels found for {label_type}")
                
                # Simple train-test split without stratification
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    codes, labels[label_type], 
                    test_size=0.2, 
                    random_state=42
                )
                
                if not train_texts or not val_texts:
                    raise ValueError(f"Empty split for {label_type}")
                
                # Log basic distribution statistics
                if wandb.run:
                    if label_type == 'key_features':
                        # For key_features, log how many samples have each feature
                        train_feature_counts = np.sum(train_labels, axis=0)
                        val_feature_counts = np.sum(val_labels, axis=0)
                        
                        feature_stats = {
                            'feature_names': list(self.label_mappings['key_features'].keys()),
                            'train_counts': train_feature_counts.tolist(),
                            'val_counts': val_feature_counts.tolist(),
                        }
                        
                        wandb.log({
                            'key_features_distribution': feature_stats,
                            'total_train_samples': len(train_texts),
                            'total_val_samples': len(val_texts),
                            'avg_features_per_sample': np.mean(np.sum(train_labels, axis=1))
                        })
                    else:
                        # For design_pattern, just log the distribution of patterns
                        pattern_names = list(self.label_mappings['design_pattern'].keys())
                        train_pattern_dist = [int(np.sum(train_labels == i)) for i in range(len(pattern_names))]
                        
                        wandb.log({
                            'design_patterns': pattern_names,
                            'pattern_distribution': dict(zip(pattern_names, train_pattern_dist))
                        })
                
                # Tokenize the texts
                train_encodings = self.tokenizer(
                    train_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512,
                    return_tensors='pt'
                )
                val_encodings = self.tokenizer(
                    val_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Set multi-label flag for key_features
                is_multilabel = (label_type == 'key_features')
                
                datasets[label_type] = {
                    'train': HDLDataset(train_encodings, train_labels, is_multilabel=is_multilabel),
                    'val': HDLDataset(val_encodings, val_labels, is_multilabel=is_multilabel)
                }
                
                # Log basic dataset info
                self.logger.info(
                    f"\nDataset created for {label_type}:"
                    f"\n - Training samples: {len(train_texts)}"
                    f"\n - Validation samples: {len(val_texts)}"
                    f"\n - {'Multiple labels per sample' if is_multilabel else 'Single label per sample'}"
                )
                
                if is_multilabel:
                    avg_features = np.mean(np.sum(train_labels, axis=1))
                    self.logger.info(f" - Average features per sample: {avg_features:.2f}")
                
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error in prepare_datasets: {str(e)}")
            traceback.print_exc()
            raise


######################################################################
######################################################################
######################################################################

    def train_models(self, output_dir: str = "hdl_models", training_params: Dict[str, int] = None, 
                    selected_type: str = None) -> None:
        """Train models with proper multi-label configuration."""
        if training_params is None:
            training_params = {
                'design_pattern': 40,
                'key_features': 30
            }

        datasets = self.prepare_datasets()
        repo_id = os.getenv('HF_REPO_ID', 'jacksonferrigno/argos-ai')
        
        # Filter types to train
        types_to_train = [selected_type] if selected_type else ['design_pattern', 'key_features']
        
        for label_type in types_to_train:
            if label_type not in datasets:
                self.logger.error(f"Invalid type: {label_type}")
                continue
                
            self.logger.info(f"\nStarting training for {label_type} with {training_params[label_type]} epochs")
            
            data = datasets[label_type]
            num_labels = len(self.label_mappings[label_type])
            
            # Use enhanced model for key_features (multi-label)
            if label_type == 'key_features':
                try:
                    # Calculate label weights for multi-label training
                    label_counts = np.sum(data['train'].labels, axis=0)
                    total_samples = len(data['train'].labels)
                    label_weights = (total_samples - label_counts) / np.maximum(label_counts, 1)
                    
                    model = create_enhanced_model(self.model_name, num_labels)
                    
                    # Optimized training arguments for key_features
                    training_args = TrainingArguments(
                        output_dir=f"{output_dir}/{label_type}",
                        num_train_epochs=training_params[label_type],
                        
                        # Batch size and gradient optimization
                        per_device_train_batch_size=16,  # Increased from 8
                        per_device_eval_batch_size=16,    # Increased from 8
                        gradient_accumulation_steps=2,   # Reduced from 4
                        max_grad_norm=0.5,              # Reduced from 1.0
                        
                        # Learning rate and schedule
                        learning_rate=5e-5,                # Increased learning rate
                        warmup_ratio=0.1,
                        lr_scheduler_type="linear",     # Changed from cosine
                        
                        # Memory optimization
                        fp16=torch.cuda.is_available(),
                        gradient_checkpointing=True,
                        
                        # Evaluation and saving
                        evaluation_strategy="steps",
                        eval_steps=100,                 # Increased from 50
                        save_strategy="steps",
                        save_steps=100,                 # Increased from 50
                        save_total_limit=2,
                        load_best_model_at_end=True,
                        metric_for_best_model="eval_loss",
                        
                        # Performance settings
                        dataloader_num_workers=2,       # Reduced from 4
                        dataloader_pin_memory=True,
                        
                        # Logging
                        logging_dir=f"{output_dir}/logs/{label_type}",
                        logging_strategy="steps",
                        logging_steps=50,
                        
                        # Misc
                        push_to_hub=False,
                        run_name=f"hdl_{label_type}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                        
                        # Added auto device detection
                        no_cuda=not torch.cuda.is_available()
                    )
                    
                    trainer = EnhancedMultilabelTrainer(
                        model=model,
                        args=training_args,
                        train_dataset=data['train'],
                        eval_dataset=data['val'],
                        focal_loss_gamma=2.0,
                        label_weights=label_weights.tolist(),
                        threshold=0.5
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error setting up key_features training: {str(e)}")
                    continue
                    
            else:
                try:
                    # Standard single-label configuration
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        num_labels=num_labels
                    )
                    
                    training_args = TrainingArguments(
                        output_dir=f"{output_dir}/{label_type}",
                        num_train_epochs=training_params[label_type],
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=8,
                        warmup_ratio=0.1,
                        weight_decay=0.01,
                        learning_rate=2e-5,
                        logging_dir=f"{output_dir}/logs/{label_type}",
                        logging_steps=100,
                        evaluation_strategy="steps",
                        eval_steps=100,
                        save_steps=100,
                        load_best_model_at_end=True,
                        gradient_checkpointing=True,
                        fp16=torch.cuda.is_available(),
                        push_to_hub=False,
                        no_cuda=not torch.cuda.is_available()
                    )
                    
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=data['train'],
                        eval_dataset=data['val']
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error setting up {label_type} training: {str(e)}")
                    continue
            
            try:
                # Enable gradient checkpointing for memory efficiency
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                
                # Train the model
                trainer.train()
                
                # Save model locally
                final_local_path = f"{output_dir}/{label_type}_final"
                trainer.save_model(final_local_path)
                self.tokenizer.save_pretrained(final_local_path)
                
                # Push to Hub if configured
                if os.getenv('HF_PUSH_TO_HUB', 'false').lower() == 'true':
                    trainer.push_to_hub(
                        repo_id=repo_id,
                        subfolder=f"hdl_models/{label_type}_final",
                        commit_message=f"Upload final {label_type} model",
                        blocking=True
                    )
                
                self.logger.info(f"Completed training for {label_type}")
                
            except Exception as e:
                self.logger.error(f"Error during {label_type} training/saving: {str(e)}")
                continue
######################################################################
######################################################################
######################################################################
def main():
    """Main execution function for model training."""
    try:
        print("\n=== Starting HDL Model Training ===\n")
        
        training_params = {
            'design_pattern': 40,  # More epochs for design patterns
            'key_features': 30     # Fewer epochs for key features since it's multi-label
        }
        
        trainer = HDLModelTrainer()
        
        # Choose which to train by uncommenting:
              
        # Train just design pattern:
        # trainer.train_models(training_params=training_params, selected_type='design_pattern')
        
        # Train just key features:
        trainer.train_models(training_params=training_params, selected_type='key_features')
        
        # Train both:
        # trainer.train_models(training_params=training_params)
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        print("\n=== HDL Model Training Complete ===\n")

if __name__ == "__main__":
    main()