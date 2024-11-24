from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer
)
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple
import json
import seaborn as sns
import matplotlib.pyplot as plt

# Import the HDLDataset and ModelLogger classes
from HDL_dataset import HDLDataset
from model_logger import ModelLogger

class HDLModelTrainer:
    """Handles training and evaluation of HDL analysis models using labeled data."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = ModelLogger().logger
        
        self.label_mappings = {
            'component_type': {},  # Will be populated from data like design_pattern
            'complexity_level': {str(i): i-1 for i in range(1, 6)},
            'design_pattern': {}   # Will be populated from data
        }
        
        self.label_mappings = {
            'component_type': {
                'StateMachine': 0, 'Counter': 1, 'ALU': 2, 
                'Package': 3, 'Interface': 4, 'Memory': 5, 
                'Controller': 6, 'Decoder': 7, 'TestBench': 8, 
                'Other': 9
            },
            'complexity_level': {str(i): i-1 for i in range(1, 6)},
            'design_pattern': {}  # Will be populated from data
        }
    def load_training_data(self) -> Tuple[List[str], Dict[str, List[int]]]:
        """Load labeled HDL code from MongoDB with proper None handling."""
        codes = []
        labels = {
            'component_type': [],
            'complexity_level': [],
            'design_pattern': []
        }
        component_types = set()
        design_patterns = set()
        
        try:
            cursor = self.collection.find({
                "analysis": {"$exists": True},
                "processing_failed": {"$ne": True},
                "content": {"$ne": None},
                "analysis.component_type": {"$exists": True},
                "analysis.complexity_level": {"$exists": True},
                "analysis.design_pattern": {"$exists": True}
            })

            # First pass: collect all unique types
            for doc in cursor:
                analysis = doc.get('analysis', {})
                
                comp_type = analysis.get('component_type')
                if comp_type and isinstance(comp_type, str):
                    component_types.add(comp_type)
                    
                pattern = analysis.get('design_pattern')
                if pattern and isinstance(pattern, str):
                    design_patterns.add(pattern)
            
            # Create mappings
            self.label_mappings['component_type'] = {
                comp_type: idx for idx, comp_type in enumerate(sorted(component_types))
                if comp_type is not None
            }
            
            self.label_mappings['design_pattern'] = {
                pattern: idx for idx, pattern in enumerate(sorted(design_patterns))
                if pattern is not None
            }
            
            # Reset cursor for second pass
            cursor.rewind()
            
            # Second pass: collect data and labels
            for doc in cursor:
                content = doc.get('content')
                analysis = doc.get('analysis', {})
                
                if not content or not isinstance(content, str):
                    continue
                    
                comp_type = analysis.get('component_type')
                complexity = str(analysis.get('complexity_level', ''))
                pattern = analysis.get('design_pattern')
                
                if not all([
                    comp_type and isinstance(comp_type, str) and comp_type in self.label_mappings['component_type'],
                    complexity and complexity.isdigit(),
                    pattern and isinstance(pattern, str) and pattern in self.label_mappings['design_pattern']
                ]):
                    continue
                
                codes.append(content)
                
                labels['component_type'].append(
                    self.label_mappings['component_type'][comp_type]
                )
                labels['complexity_level'].append(
                    self.label_mappings['complexity_level'].get(complexity, 2)
                )
                labels['design_pattern'].append(
                    self.label_mappings['design_pattern'][pattern]
                )
            
            if not codes or not all(len(l) == len(codes) for l in labels.values()):
                raise ValueError("No valid training data found or label lengths mismatch")
                
            print(f"Loaded {len(codes)} valid training samples")
            print(f"Component types found: {len(component_types)}")
            print(f"Design patterns found: {len(design_patterns)}")
            print(f"Label counts: {[len(l) for l in labels.values()]}")
            
            with open('label_mappings.json', 'w') as f:
                json.dump(self.label_mappings, f, indent=2)
            
            return codes, labels
            
        except Exception as e:
            print(f"Error in load_training_data: {str(e)}")
            print("Stacktrace:", exc_info=True)
            raise


    def prepare_datasets(self) -> Dict[str, Dict[str, HDLDataset]]:
        """Prepare training and validation datasets with error handling."""
        try:
            codes, labels = self.load_training_data()
            if not codes:
                raise ValueError("No training data available")
                
            datasets = {}
            
            for label_type in ['component_type', 'complexity_level', 'design_pattern']:
                if not labels[label_type]:
                    raise ValueError(f"No labels found for {label_type}")
                    
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    codes, labels[label_type], test_size=0.2, random_state=42
                )
                
                # Validate data before tokenization
                if not train_texts or not val_texts:
                    raise ValueError(f"Empty split for {label_type}")
                
                train_encodings = self.tokenizer(
                    train_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                val_encodings = self.tokenizer(
                    val_texts, 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                
                datasets[label_type] = {
                    'train': HDLDataset(train_encodings, train_labels),
                    'val': HDLDataset(val_encodings, val_labels)
                }
                
            return datasets
            
        except Exception as e:
            print(f"Error in prepare_datasets: {str(e)}")
            print("Stacktrace:", exc_info=True)
            raise

    def train_models(self, output_dir: str = "hdl_models", training_params: Dict[str, int] = None, 
                    selected_type: str = None) -> None:
        """
        Train models selectively or all at once.
        
        Args:
            output_dir: Directory to save models
            training_params: Dictionary with epochs for each type
            selected_type: Optional specific type to train ('component_type', 'complexity_level', or 'design_pattern')
        """
        if training_params is None:
            training_params = {
                'component_type': 10,
                'complexity_level': 10,
                'design_pattern': 30
            }

        datasets = self.prepare_datasets()
        
        # Filter types to train
        types_to_train = [selected_type] if selected_type else ['component_type', 'complexity_level', 'design_pattern']
        
        for label_type in types_to_train:
            if label_type not in datasets:
                self.logger.error(f"Invalid type: {label_type}")
                continue
                
            self.logger.info(f"\nStarting training for {label_type} with {training_params[label_type]} epochs")
            
            data = datasets[label_type]
            num_labels = len(self.label_mappings[label_type])
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=num_labels
            )
            
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/{label_type}",
                num_train_epochs=training_params[label_type],
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs/{label_type}",
                logging_steps=100,
                evaluation_strategy="steps",
                eval_steps=100,
                save_steps=100,
                load_best_model_at_end=True,
                learning_rate=2e-5,
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=data['train'],
                eval_dataset=data['val']
            )
            
            trainer.train()
            
            # Save the model
            model_path = f"{output_dir}/{label_type}_final"
            trainer.save_model(model_path)
            self.tokenizer.save_pretrained(model_path)
            
            self.logger.info(f"Completed training for {label_type}")



def main():
    """Main execution function for model training."""
    try:
        print("\n=== Starting HDL Model Training ===\n")
        
        training_params = {
            'component_type': 10,
            'complexity_level': 10,
            'design_pattern': 40  # Increased epochs for design pattern
        }
        
        trainer = HDLModelTrainer()
        
        # Choose which to train by uncommenting:
        
        # Train all models:
        # trainer.train_models(training_params=training_params)
        
        # Train just design pattern:
        trainer.train_models(training_params=training_params, selected_type='design_pattern')
        
        # Train just component type:
        # trainer.train_models(training_params=training_params, selected_type='component_type')
        
        # Train just complexity:
        # trainer.train_models(training_params=training_params, selected_type='complexity_level')
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== HDL Model Training Complete ===\n")

if __name__ == "__main__":
    main()