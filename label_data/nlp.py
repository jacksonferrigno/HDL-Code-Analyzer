import warnings
import logging
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig  # Important for HF model saving
)
import os
from dotenv import load_dotenv
import json
import traceback

from utils.vhdl_segmenter import VHDLSegmenter
from utils.model_logger import ModelLogger
from utils.HDL_dataset import HDLDataset
from utils.db_handler import HDLDatabaseHandler

class HDLSegmentTrainer:
    """Trains models for analyzing VHDL code segments and patterns.
    
    This trainer handles both segment-level features and full-code design patterns,
    managing the entire pipeline from data loading through training and model saving."""
    
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        """Initialize trainer with model and connections."""


        # Initialize connections and base components
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        # Verify HuggingFace token exists
        if not os.getenv('HF_TOKEN'):
            raise ValueError("HF_TOKEN environment variable required for model saving")
        
        # Set up model components
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = ModelLogger()
        self.segmenter = VHDLSegmenter()
        self.db_handler = HDLDatabaseHandler()
        
        # Classification mappings
        self.feature_mapping = {}  # For segment features
        self.pattern_mapping = {}  # For design patterns
        self.component_mapping = {}  # For component types

    def train_model(self, model_type: str, output_dir: str = "hdl_models"):
        """Train model on common HDL code features."""
        try:
            # Use existing data loading
            texts, labels, segment_types = self.db_handler.load_segment_data()
            num_labels = len(self.feature_mapping)
            print(f"\nStarting training for {num_labels} common features")
            
            # Split with stratification
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels,
                test_size=0.2,
                random_state=42,
                stratify=labels  # Ensure balanced split
            )
            print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
            
            # Keep existing data preparation
            train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512)
            val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=512)
            
            train_dataset = HDLDataset(train_encodings, train_labels)
            val_dataset = HDLDataset(val_encodings, val_labels)
            
            # Keep existing model initialization
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels
            )
            
            # Update only the training arguments
            training_args = TrainingArguments(
                output_dir=f"{output_dir}/{model_type}",
                
                # Hub settings
                push_to_hub=True,
                hub_model_id=f"hdl-{model_type}-classifier",
                hub_strategy="checkpoint",
                
                # Core training parameters
                learning_rate=3e-5,
                num_train_epochs=12,
                per_device_train_batch_size=32,
                gradient_accumulation_steps=2,
                
                # Learning rate schedule
                lr_scheduler_type="linear",
                warmup_ratio=0.1,
                
                # Regularization (simplified but effective)
                weight_decay=0.01,
                label_smoothing_factor=0.1,
                max_grad_norm=1.0,
                
                # Evaluation
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=3,
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                greater_is_better=True,
                
                # Performance
                fp16=True,
                dataloader_num_workers=4,
                
                # Logging
                logging_steps=50,
                report_to=["tensorboard"],
            )
            
            # Keep existing metrics computation
            def compute_metrics(eval_pred):
                predictions, labels = eval_pred
                predictions = np.argmax(predictions, axis=1)
                return {
                    'accuracy': accuracy_score(labels, predictions),
                    'f1': f1_score(labels, predictions, average='weighted')
                }
            
            # Keep existing trainer setup with early stopping
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            print("\nTraining started...")
            trainer.train()
            
            # Keep existing model saving
            final_path = f"{output_dir}/{model_type}_final"
            trainer.save_model(final_path)
            self.tokenizer.save_pretrained(final_path)
            
            # Keep existing mapping saving
            mapping_path = os.path.join(final_path, "feature_mapping.json")
            with open(mapping_path, 'w') as f:
                json.dump(self.feature_mapping, f, indent=2)
            
            print(f"\nTraining complete. Model and mappings saved to {final_path}")
            
        except Exception as e:
            print(f"\nError during training: {str(e)}")
            traceback.print_exc()

    def close(self):
        """Close any resources if necessary."""
        self.db_handler.close()

def main():
    """Main execution function."""
    try:
        print("\n=== Starting HDL Model Training ===")
        
        trainer = HDLSegmentTrainer()
        
        # Train both classifiers
        trainer.train_model('segment')
        #trainer.train_model('pattern')
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        if 'trainer' in locals():
            trainer.close()
        print("\n=== HDL Model Training Complete ===")

if __name__ == "__main__":
    main()