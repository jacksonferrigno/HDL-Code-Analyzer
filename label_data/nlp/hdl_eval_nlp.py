import os
from dotenv import load_dotenv
from pymongo import MongoClient
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
import traceback

class HDLEvaluator:
    """Handles evaluation of HDL analysis models."""
    
    def __init__(self):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable must be set for private repo access")

        self.base_model = "microsoft/codebert-base"
        self.repo_id = "jacksonferrigno/argos-ai"
        self.model_types = ['design_pattern', 'key_features']
        
        self.models = {}
        self.tokenizer = None
        self.load_trained_models()
        
    def load_trained_models(self):
        """Load pre-trained models with proper error handling."""
        try:
            with open('label_mappings.json', 'r') as f:
                self.label_mappings = json.load(f)
            print("Successfully loaded label mappings")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                token=self.hf_token
            )
            print("Successfully loaded tokenizer")

            for model_type in self.model_types:
                try:
                    print(f"\nAttempting to load {model_type} model...")
                    subfolder = f"logs/hdl_models/{model_type}_final"
                    model_path = f"{self.repo_id}"
                    
                    config = AutoConfig.from_pretrained(
                        model_path,
                        token=self.hf_token,
                        subfolder=subfolder,
                        trust_remote_code=True,
                        local_files_only=False,
                        revision="main"
                    )
                    
                    if model_type == 'key_features':
                        config.num_labels = len(self.label_mappings[model_type])
                        config.problem_type = 'multi_label_classification'
                    else:
                        config.problem_type = 'single_label_classification'
                    
                    print(f"Model configured for {config.num_labels} labels")
                    
                    self.models[model_type] = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        config=config,
                        token=self.hf_token,
                        trust_remote_code=True,
                        subfolder=subfolder,
                        local_files_only=False,
                        revision="main",
                        ignore_mismatched_sizes=True
                    )
                    print(f"Successfully loaded {model_type} model")
                    
                except Exception as e:
                    print(f"Error loading {model_type} model: {str(e)}")
                    traceback.print_exc()
                    continue
            
            if not self.models:
                raise ValueError("No models were successfully loaded")
            
            loaded_models = ", ".join(self.models.keys())
            print(f"\nSuccessfully loaded models: {loaded_models}")
                
        except Exception as e:
            print("Fatal error in model loading:")
            traceback.print_exc()
            raise

    def analyze_single(self, hdl_code: str, top_n: int = 3) -> Dict[str, Any]:
        """Analyze a single piece of HDL code with proper error handling."""
        try:
            encodings = self.tokenizer(
                hdl_code,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            results = {}
            
            for model_type, model in self.models.items():
                try:
                    outputs = model(**encodings)
                    
                    if model_type == 'key_features':
                        logits = outputs.logits
                        predictions = torch.sigmoid(logits)
                        predictions_np = predictions.squeeze().detach().numpy()
                        
                        # Get indices of top N predictions
                        top_indices = np.argsort(predictions_np)[-top_n:][::-1]
                        confidences = predictions_np[top_indices]
                        
                        # Map indices to feature names
                        reverse_mapping = {v: k for k, v in self.label_mappings[model_type].items()}
                        predicted_features = []
                        feature_confidences = []
                        
                        for idx, conf in zip(top_indices, confidences):
                            if idx in reverse_mapping:
                                feature = reverse_mapping[idx]
                                predicted_features.append(feature)
                                feature_confidences.append({
                                    'feature': feature,
                                    'confidence': float(conf)
                                })
                        
                        results[model_type] = {
                            'predictions': predicted_features,
                            'confidences': confidences.tolist(),
                            'feature_confidences': feature_confidences
                        }
                    else:
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(predictions, dim=-1).item()
                        confidence = float(torch.max(predictions))
                        
                        reverse_mapping = {v: k for k, v in self.label_mappings[model_type].items()}
                        if predicted_class in reverse_mapping:
                            results[model_type] = {
                                'prediction': reverse_mapping[predicted_class],
                                'confidence': confidence
                            }
                        else:
                            results[model_type] = {
                                'error': f'Invalid prediction index: {predicted_class}'
                            }
                    
                except Exception as e:
                    print(f"Error in prediction for {model_type}: {str(e)}")
                    traceback.print_exc()
                    results[model_type] = {'error': str(e)}
            
            return results
                
        except Exception as e:
            print(f"Error in analyze_single: {str(e)}")
            traceback.print_exc()
            return {'error': str(e)}

    def evaluate_batch(self, sample_size: int = 100, top_n: int = 3) -> Dict[str, Any]:
        """Evaluate models on a batch of documents with proper error handling."""
        print(f"\nStarting batch evaluation on {sample_size} samples, returning top {top_n} features...")
        
        test_data = list(self.collection.find({
            "analysis": {"$exists": True},
            "analysis.key_features": {"$exists": True},
            "processing_failed": {"$ne": True}
        }).limit(sample_size))
        
        print(f"Retrieved {len(test_data)} documents for evaluation")
        
        results = {
            'design_pattern': {'correct': 0, 'total': 0, 'predictions': []},
            'key_features': {'correct': 0, 'total': 0, 'predictions': [], 'recall_at_n': 0},
            'examples': [],
            'errors': []
        }
        
        for index, doc in enumerate(test_data, 1):
            try:
                if index % 10 == 0:
                    print(f"Processing document {index}/{len(test_data)}")
                    
                original = doc['analysis']
                predicted = self.analyze_single(doc['content'], top_n=top_n)
                
                if not predicted or 'error' in predicted:
                    results['errors'].append({
                        'doc_id': str(doc.get('_id')),
                        'error': predicted.get('error', 'Unknown error')
                    })
                    continue
                
                for model_type in self.models.keys():
                    if model_type not in original or model_type not in predicted:
                        continue
                        
                    if 'error' in predicted[model_type]:
                        continue
                        
                    if model_type == 'key_features':
                        true_features = original[model_type]
                        if not isinstance(true_features, list):
                            true_features = [true_features] if true_features else []
                        
                        pred_features = predicted[model_type].get('predictions', [])
                        correct_features = set(pred_features).intersection(set(true_features))
                        
                        results[model_type]['total'] += len(true_features)
                        results[model_type]['correct'] += len(correct_features)
                        
                        results[model_type]['predictions'].append({
                            'expected': true_features,
                            'predicted': pred_features,
                            'feature_confidences': predicted[model_type].get('feature_confidences', []),
                            'correct_features': list(correct_features),
                            'recall': len(correct_features) / len(true_features) if true_features else 0
                        })
                    else:
                        results[model_type]['total'] += 1
                        pred_value = predicted[model_type]['prediction']
                        if str(original[model_type]) == str(pred_value):
                            results[model_type]['correct'] += 1
                        
                        results[model_type]['predictions'].append({
                            'expected': original[model_type],
                            'predicted': pred_value,
                            'confidence': predicted[model_type]['confidence'],
                            'correct': str(original[model_type]) == str(pred_value)
                        })
                
                results['examples'].append({
                    'content_preview': doc['content'][:200],
                    'original': original,
                    'predicted': predicted
                })
                
            except Exception as e:
                print(f"Error processing document {index}: {str(e)}")
                results['errors'].append({
                    'doc_id': str(doc.get('_id')),
                    'error': str(e)
                })
                continue
        
        # Calculate metrics
        for model_type in self.models.keys():
            if results[model_type]['total'] > 0:
                if model_type == 'key_features':
                    recall_sum = sum(p['recall'] for p in results[model_type]['predictions'])
                    results[model_type]['recall_at_n'] = recall_sum / len(results[model_type]['predictions'])
                    results[model_type]['accuracy'] = results[model_type]['correct'] / results[model_type]['total']
                else:
                    results[model_type]['accuracy'] = results[model_type]['correct'] / results[model_type]['total']
        
        return results

    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        print("\n=== HDL Model Evaluation Report ===\n")
        
        for model_type in self.models.keys():
            print(f"\n{model_type.replace('_', ' ').title()} Results:")
            if model_type == 'key_features':
                recall = results[model_type]['recall_at_n'] * 100
                print(f"Recall@N: {recall:.2f}%")
                print(f"Feature Detection Rate: {results[model_type]['accuracy'] * 100:.2f}%")
            else:
                accuracy = results[model_type]['accuracy'] * 100
                total = results[model_type]['total']
                correct = results[model_type]['correct']
                print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        print("\nSample Predictions:")
        for i, example in enumerate(results['examples'][:5]):
            print(f"\nExample {i+1}:")
            print(f"Content Preview: {example['content_preview']}...")
            
            for model_type in self.models.keys():
                if model_type in example['predicted']:
                    print(f"\n{model_type.replace('_', ' ').title()}:")
                    if model_type == 'key_features':
                        print(f"  Expected: {', '.join(example['original'][model_type])}")
                        print("  Predicted (with confidences):")
                        for fc in example['predicted'][model_type]['feature_confidences']:
                            print(f"    - {fc['feature']}: {fc['confidence']:.2f}")
                    else:
                        print(f"  Expected: {example['original'][model_type]}")
                        print(f"  Predicted: {example['predicted'][model_type]['prediction']}")
                        print(f"  Confidence: {example['predicted'][model_type]['confidence']:.2f}")
def main():
    """Main evaluation function."""
    try:
        print("\n=== Starting HDL Model Evaluation ===")
        evaluator = HDLEvaluator()
        
        # Run batch evaluation on samples from MongoDB
        sample_size = 15
        print(f"\nRunning batch evaluation on {sample_size} samples from MongoDB...")
        results = evaluator.evaluate_batch(sample_size=sample_size)
        evaluator.print_evaluation_report(results)
        
        # Save timestamp for this evaluation run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'evaluation_results_{timestamp}.json'
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nDetailed results saved to: {save_path}")
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()