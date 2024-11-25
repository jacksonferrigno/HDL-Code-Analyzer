import os
from dotenv import load_dotenv
from pymongo import MongoClient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class HDLEvaluator:
    """Handles evaluation of HDL analysis models."""
    
    def __init__(self):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        # HF auth token required for private repo
        self.hf_token = os.getenv('HF_TOKEN')
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable must be set for private repo access")

        # Base model architecture and repo settings
        self.base_model = "microsoft/codebert-base"
        self.repo_id = "jacksonferrigno/argos-ai"  # Updated with username prefix
        self.model_types = ['design_pattern', 'key_features']
        
        # Initialize models and configurations
        self.models = {}
        self.tokenizer = None
        
        # Load trained models and configurations
        self.load_trained_models()
        
    def load_trained_models(self):
        """Load pre-trained models from Hugging Face Hub."""
        try:
            # Load label mappings from local file
            with open('label_mappings.json', 'r') as f:
                self.label_mappings = json.load(f)
            print("Successfully loaded label mappings")
            
            # Load base tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                token=self.hf_token  # Add token here as well
            )
            print("Successfully loaded tokenizer")

            # Load each model type
            for model_type in self.model_types:
                try:
                    print(f"\nAttempting to load {model_type} model...")
                    
                    # Create model-specific configuration
                    config = AutoConfig.from_pretrained(
                        self.base_model,
                        num_labels=len(self.label_mappings[model_type]),
                        token=self.hf_token  # Add token here
                    )
                    
                    # Set required configuration attributes for RoBERTa
                    config.model_type = "roberta"
                    config.architectures = ["RobertaForSequenceClassification"]
                    
                    # Configure problem type for each model
                    if model_type == 'key_features':
                        config.problem_type = 'multi_label_classification'
                    else:
                        config.problem_type = 'single_label_classification'
                    
                    print(f"Successfully created config for {model_type}")
                    
                    # Load the model with the RoBERTa config
                    subfolder = f"logs/hdl_models/{model_type}_final"
                    model_path = f"{self.repo_id}"  # Use the full repo ID
                    print(f"Loading from repo: {model_path}, subfolder: {subfolder}")
                    
                    self.models[model_type] = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        config=config,
                        token=self.hf_token,  # Add token here
                        trust_remote_code=True,
                        subfolder=subfolder,
                        local_files_only=False,
                        revision="main"
                    )
                    print(f"Successfully loaded {model_type} model")
                    
                except Exception as e:
                    print(f"Error loading {model_type} model: {str(e)}")
                    print(f"Detailed error for {model_type}:")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Verify models loaded successfully
            if not self.models:
                raise ValueError("No models were successfully loaded")
            
            loaded_models = ", ".join(self.models.keys())
            print(f"\nSuccessfully loaded models: {loaded_models}")
                
        except Exception as e:
            print("Fatal error in model loading:")
            import traceback
            traceback.print_exc()
            raise
    def analyze_single(self, hdl_code: str) -> Dict[str, Any]:
        """Analyze a single piece of HDL code with fixed multi-label handling."""
        try:
            encodings = self.tokenizer(
                hdl_code,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            results = {}
            
            # Process with each loaded model
            for model_type, model in self.models.items():
                try:
                    outputs = model(**encodings)
                    
                    if model_type == 'key_features':
                        # Multi-label prediction handling
                        logits = outputs.logits
                        predictions = torch.sigmoid(logits)  # Use sigmoid for multi-label
                        threshold = 0.5
                        
                        # Get predictions above threshold
                        predicted_classes = (predictions > threshold).squeeze().nonzero().flatten().tolist()
                        confidences = predictions.squeeze()[predicted_classes].tolist()
                        
                        # Map indices back to feature names
                        reverse_mapping = {v: k for k, v in self.label_mappings[model_type].items()}
                        predicted_features = [reverse_mapping[idx] for idx in predicted_classes]
                        
                        results[model_type] = {
                            'predictions': predicted_features,
                            'confidences': confidences if isinstance(confidences, list) else [confidences]
                        }
                    else:
                        # Single-label prediction (unchanged)
                        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                        predicted_class = torch.argmax(predictions, dim=-1).item()
                        confidence = float(torch.max(predictions))
                        
                        reverse_mapping = {v: k for k, v in self.label_mappings[model_type].items()}
                        results[model_type] = {
                            'prediction': reverse_mapping[predicted_class],
                            'confidence': confidence
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

    def evaluate_batch(self, sample_size: int = 100) -> Dict[str, Any]:
        """Evaluate models on a batch of existing analyzed documents."""
        print(f"\nStarting batch evaluation on {sample_size} samples...")
        
        test_data = list(self.collection.find({
            "analysis": {"$exists": True},
            "processing_failed": {"$ne": True}
        }).limit(sample_size))
        
        print(f"Retrieved {len(test_data)} documents for evaluation")
        
        results = {
            'design_pattern': {'correct': 0, 'total': 0, 'predictions': []},
            'key_features': {'correct': 0, 'total': 0, 'predictions': []},
            'examples': []
        }
        
        for index, doc in enumerate(test_data, 1):
            if index % 10 == 0:
                print(f"Processing document {index}/{len(test_data)}")
                
            original = doc['analysis']
            predicted = self.analyze_single(doc['content'])
            
            if predicted and 'error' not in predicted:
                # Add to results
                for model_type in self.models.keys():
                    if model_type in original and model_type in predicted:
                        results[model_type]['total'] += 1
                        
                        if model_type == 'key_features':
                            # Handle multi-label case
                            pred_features = set(predicted[model_type]['predictions'])
                            true_features = set(original[model_type])
                            correct_features = pred_features.intersection(true_features)
                            results[model_type]['correct'] += len(correct_features)
                            
                            results[model_type]['predictions'].append({
                                'expected': list(true_features),
                                'predicted': list(pred_features),
                                'confidences': predicted[model_type]['confidences']
                            })
                        else:
                            # Handle single-label case
                            pred_value = predicted[model_type]['prediction']
                            if str(original[model_type]) == str(pred_value):
                                results[model_type]['correct'] += 1
                            
                            results[model_type]['predictions'].append({
                                'expected': original[model_type],
                                'predicted': pred_value,
                                'confidence': predicted[model_type]['confidence'],
                                'correct': str(original[model_type]) == str(pred_value)
                            })
                
                # Store detailed example
                results['examples'].append({
                    'content_preview': doc['content'][:200],
                    'original': original,
                    'predicted': predicted
                })
        
        # Calculate accuracies
        for model_type in self.models.keys():
            if results[model_type]['total'] > 0:
                if model_type == 'key_features':
                    total_features = sum(len(p['expected']) for p in results[model_type]['predictions'])
                    accuracy = results[model_type]['correct'] / total_features if total_features > 0 else 0
                else:
                    accuracy = results[model_type]['correct'] / results[model_type]['total']
                results[model_type]['accuracy'] = accuracy
        
        return results

    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        print("\n=== HDL Model Evaluation Report ===\n")
        
        for model_type in self.models.keys():
            print(f"\n{model_type.replace('_', ' ').title()} Results:")
            if 'accuracy' in results[model_type]:
                accuracy = results[model_type]['accuracy'] * 100
                total = results[model_type]['total']
                if model_type == 'key_features':
                    print(f"Average Feature Detection Rate: {accuracy:.2f}%")
                else:
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
                        print(f"  Predicted: {', '.join(example['predicted'][model_type]['predictions'])}")
                        for feat, conf in zip(example['predicted'][model_type]['predictions'],
                                           example['predicted'][model_type]['confidences']):
                            print(f"    - {feat}: {conf:.2f}")
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