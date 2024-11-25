from transformers import AutoConfig, AutoModelForSequenceClassification, Trainer
import torch
from torch import nn
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Optional, Union, Tuple, Any
import wandb
import torch.nn.functional as F

class EnhancedMultilabelTrainer(Trainer):
    """Enhanced trainer for multi-label classification with improved metrics and loss functions."""
    
    def __init__(self, *args, **kwargs):
        self.focal_loss_gamma = kwargs.pop('focal_loss_gamma', 2.0)
        self.label_weights = kwargs.pop('label_weights', None)
        self.threshold = kwargs.pop('threshold', 0.5)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute the training loss."""
        try:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply label smoothing
            if labels is not None:
                labels = labels.float() * 0.9 + 0.05
            
            # Weighted Focal Loss implementation
            if self.label_weights is not None:
                weights = torch.tensor(self.label_weights).to(logits.device)
            else:
                pos_weight = self.calculate_class_weights(labels)
                weights = pos_weight.to(logits.device)
            
            # Compute base BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.model.config.num_labels),
                labels.float().view(-1, self.model.config.num_labels),
                pos_weight=weights,
                reduction='none'
            )
            
            # Apply Focal Loss modulation
            pt = torch.exp(-bce_loss)
            focal_loss = ((1 - pt) ** self.focal_loss_gamma) * bce_loss
            
            # Add L1 regularization for sparsity
            l1_lambda = 0.01
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            
            loss = focal_loss.mean() + l1_lambda * l1_reg
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            print(f"Error in compute_loss: {str(e)}")
            raise

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                     num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Args:
            model: The model to train
            inputs: Dict of input tensors
            num_items_in_batch: Number of items in current batch 

        Returns:
            torch.Tensor: The training loss
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1 and (num_items_in_batch is not None):
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

    def prediction_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], 
                       prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None) -> Tuple[Optional[torch.Tensor], ...]:
        """Perform an evaluation step."""
        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
            # Temperature scaling for probability calibration
            temperature = 1.5
            logits = outputs.logits / temperature
            
            # Convert logits to probabilities
            probs = torch.sigmoid(logits)
            
            if prediction_loss_only:
                return (loss, None, None)

            labels = None
            if has_labels:
                labels = inputs["labels"]
                if labels.shape[-1] != logits.shape[-1]:
                    labels = labels.view(-1, logits.shape[-1])

            return (loss, probs, labels)

    def calculate_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Calculate class weights based on label distribution."""
        pos_counts = torch.sum(labels, dim=0)
        neg_counts = labels.size(0) - pos_counts
        pos_weight = neg_counts / torch.clamp(pos_counts, min=1)
        return pos_weight

    def compute_metrics(self, eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Apply sigmoid and threshold
        sigmoid_preds = 1 / (1 + np.exp(-predictions))
        binary_preds = (sigmoid_preds > self.threshold).astype(float)
        
        metrics = {}
        
        try:
            # Sample-averaged metrics
            metrics['macro_f1'] = f1_score(labels, binary_preds, average='macro', zero_division=0)
            metrics['micro_f1'] = f1_score(labels, binary_preds, average='micro', zero_division=0)
            metrics['samples_f1'] = f1_score(labels, binary_preds, average='samples', zero_division=0)
            
            # Per-class metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, binary_preds, average=None, zero_division=0
            )
            
            # ROC AUC scores
            try:
                metrics['macro_auc'] = roc_auc_score(labels, sigmoid_preds, average='macro')
                metrics['micro_auc'] = roc_auc_score(labels, sigmoid_preds, average='micro')
            except ValueError:
                metrics['macro_auc'] = 0.0
                metrics['micro_auc'] = 0.0
            
            metrics['exact_match'] = (binary_preds == labels).all(axis=1).mean()
            metrics['hamming_loss'] = (binary_preds != labels).mean()
            
            if wandb.run is not None:
                wandb.log(metrics)
            
        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            metrics = {
                'error': -1,
                'error_message': str(e)
            }
            
        return metrics

def create_enhanced_model(model_name: str, num_labels: int) -> AutoModelForSequenceClassification:
    """Create an enhanced model with improved architecture for multi-label classification."""
    try:
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type='multi_label_classification',
            hidden_dropout_prob=0.2,
            attention_probs_dropout_prob=0.1,
            hidden_act='gelu',
            layer_norm_eps=1e-7,
            classifier_dropout=0.3
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Add label correlation awareness
        correlation_matrix = nn.Parameter(torch.eye(num_labels))
        setattr(model, 'label_correlations', correlation_matrix)
        
        # Initialize weights with improved scheme
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        
        model.apply(init_weights)
        
        return model
        
    except Exception as e:
        print(f"Error in create_enhanced_model: {str(e)}")
        raise