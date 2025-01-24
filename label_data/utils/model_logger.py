import logging
import os
from datetime import datetime
from typing import Dict, Any, List
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class ModelLogger:
    """Custom logger for model training and evaluation."""
    
    def __init__(self, log_dir: str = "model_logs"):
        """Initialize logger with directory for logs and visualizations."""
        self.log_dir = log_dir
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"training_{self.run_id}")
        os.makedirs(self.log_path, exist_ok=True)
        
        # Set up file logger
        self.logger = logging.getLogger(f"HDLModel_{self.run_id}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(os.path.join(self.log_path, "training.log"))
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        # Training metrics storage
        self.metrics = {
            'training_loss': [],
            'validation_loss': [],
            'component_accuracy': [],
            'complexity_accuracy': [],
            'pattern_accuracy': []
        }
        
        self.logger.info(f"Initialized new training run: {self.run_id}")

    def log_data_stats(self, data_stats: Dict[str, Any]):
        """Log statistics about the training data."""
        self.logger.info("\n=== Training Data Statistics ===")
        self.logger.info(f"Total samples: {data_stats['total_samples']}")
        
        # Log class distribution
        for label_type, distribution in data_stats['class_distribution'].items():
            self.logger.info(f"\n{label_type} distribution:")
            for class_name, count in distribution.items():
                self.logger.info(f"  {class_name}: {count}")
        
        # Save distribution plots
        self._plot_distributions(data_stats['class_distribution'])

    def _plot_distributions(self, distributions: Dict[str, Dict[str, int]]):
        """Create and save distribution plots."""
        for label_type, dist in distributions.items():
            plt.figure(figsize=(10, 6))
            sns.barplot(x=list(dist.keys()), y=list(dist.values()))
            plt.title(f"{label_type} Distribution")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_path, f"{label_type}_distribution.png"))
            plt.close()

    def log_training_step(self, step: int, metrics: Dict[str, float], label_type: str):
        """Log training step metrics."""
        self.logger.info(f"\n=== Training Step {step} for {label_type} ===")
        self.logger.info(f"Training loss: {metrics['train_loss']:.4f}")
        self.logger.info(f"Validation loss: {metrics.get('eval_loss', 'N/A')}")
        
        # Store metrics
        self.metrics['training_loss'].append(metrics['train_loss'])
        if 'eval_loss' in metrics:
            self.metrics['validation_loss'].append(metrics['eval_loss'])

    def log_evaluation(self, label_type: str, true_labels: List[int], 
                      predicted_labels: List[int], label_names: List[str]):
        """Log evaluation metrics and confusion matrix."""
        self.logger.info(f"\n=== Evaluation Results for {label_type} ===")
        
        # Classification report
        report = classification_report(true_labels, predicted_labels, 
                                    target_names=label_names, digits=4)
        self.logger.info(f"\nClassification Report:\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        self._plot_confusion_matrix(cm, label_names, label_type)

    def _plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], label_type: str):
        """Create and save confusion matrix visualization."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {label_type}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path, f"{label_type}_confusion_matrix.png"))
        plt.close()

    def log_model_analysis(self, hdl_code: str, predictions: Dict[str, Any], 
                          confidence_scores: Dict[str, float]):
        """Log model predictions and reasoning."""
        self.logger.info("\n=== Model Analysis ===")
        self.logger.info(f"\nInput Code Preview:\n{hdl_code[:200]}...")
        
        for label_type, pred in predictions.items():
            self.logger.info(f"\n{label_type}:")
            self.logger.info(f"Prediction: {pred}")
            self.logger.info(f"Confidence: {confidence_scores[label_type]:.4f}")

    def save_run_summary(self):
        """Save summary of the training run."""
        summary = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'final_metrics': {
                'training_loss': self.metrics['training_loss'][-1],
                'validation_loss': self.metrics['validation_loss'][-1] 
                    if self.metrics['validation_loss'] else None,
                'component_accuracy': self.metrics['component_accuracy'][-1] 
                    if self.metrics['component_accuracy'] else None
            }
        }
        
        with open(os.path.join(self.log_path, 'run_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        self._plot_training_curves()

    def _plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['training_loss'], label='Training Loss')
        if self.metrics['validation_loss']:
            plt.plot(self.metrics['validation_loss'], label='Validation Loss')
        plt.title('Training Curves')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_path, 'training_curves.png'))
        plt.close()

    def log_error(self, error_msg: str, error_type: str):
        """Log errors with detailed information."""
        self.logger.error(f"\n=== {error_type} Error ===")
        self.logger.error(error_msg)
        self.logger.error("Stack trace:", exc_info=True)

