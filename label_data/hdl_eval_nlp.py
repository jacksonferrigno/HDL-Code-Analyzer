
import os
from dotenv import load_dotenv
from pymongo import MongoClient
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from datetime import datetime
from typing import Dict, Any, List, Optional

class HDLEvaluator:
    """Separate evaluator for trained HDL models."""
    
    def __init__(self):
        load_dotenv()
        self.mongo_client = MongoClient(os.getenv('DB_URI'))
        self.db = self.mongo_client['hdl_database']
        self.collection = self.db['hdl_codes']
        
        # Load trained models and configurations
        self.load_trained_models()
        
    def load_trained_models(self):
        """Load pre-trained models and configurations."""
        try:
            # Load label mappings
            with open('label_mappings.json', 'r') as f:
                self.label_mappings = json.load(f)
            
            # Load models
            self.models = {}
            self.tokenizer = None
            
            for label_type in ['component_type', 'complexity_level', 'design_pattern']:
                model_path = f"hdl_models/{label_type}_final"
                if os.path.exists(model_path):
                    self.models[label_type] = AutoModelForSequenceClassification.from_pretrained(model_path)
                    if not self.tokenizer:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                    print(f"Loaded model for {label_type}")
                else:
                    raise ValueError(f"No trained model found for {label_type}")
                    
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def analyze_single(self, hdl_code: str) -> Dict[str, Any]:
        """Analyze a single piece of HDL code."""
        try:
            with open('label_mappings.json', 'r') as f:
                label_mappings = json.load(f)
            
            encodings = self.tokenizer(
                hdl_code,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            results = {}
            
            for label_type in ['component_type', 'complexity_level', 'design_pattern']:
                # Updated to handle safetensors format
                model_path = f"hdl_models/{label_type}_final"
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True  # Ensure it uses local files
                )
                
                outputs = model(**encodings)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                
                reverse_mapping = {v: k for k, v in label_mappings[label_type].items()}
                results[label_type] = {
                    'prediction': reverse_mapping[predicted_class],
                    'confidence': float(torch.max(predictions))
                }
            
            return results
            
        except Exception as e:
            print(f"Error analyzing HDL code: {str(e)}")
            return None

    def evaluate_batch(self, sample_size: int = 50) -> Dict[str, Any]:
        """Evaluate model on a batch of existing analyzed documents."""
        test_data = list(self.collection.find({
            "analysis": {"$exists": True},
            "processing_failed": {"$ne": True}
        }).limit(sample_size))
        
        results = {
            'component_type': {'correct': 0, 'total': 0, 'predictions': []},
            'complexity_level': {'correct': 0, 'total': 0, 'predictions': []},
            'design_pattern': {'correct': 0, 'total': 0, 'predictions': []},
            'examples': []
        }
        
        for doc in test_data:
            original = doc['analysis']
            predicted = self.analyze_single(doc['content'])
            
            if predicted:
                for field in self.models.keys():
                    if field in original:
                        results[field]['total'] += 1
                        pred_value = predicted[field]['prediction']
                        if str(original[field]) == str(pred_value):
                            results[field]['correct'] += 1
                        
                        results[field]['predictions'].append({
                            'expected': original[field],
                            'predicted': pred_value,
                            'confidence': predicted[field]['confidence'],
                            'correct': str(original[field]) == str(pred_value)
                        })
                
                # Store detailed example
                results['examples'].append({
                    'content_preview': doc['content'][:200],
                    'original': original,
                    'predicted': predicted
                })
        
        # Calculate accuracies
        for field in self.models.keys():
            if results[field]['total'] > 0:
                results[field]['accuracy'] = results[field]['correct'] / results[field]['total']
        
        # Save evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'evaluation_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

    def print_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        print("\n=== HDL Model Evaluation Report ===\n")
        
        print("Overall Accuracy:")
        for field in self.models.keys():
            accuracy = results[field]['accuracy'] * 100
            total = results[field]['total']
            correct = results[field]['correct']
            print(f"{field}: {accuracy:.2f}% ({correct}/{total})")
        
        print("\nSample Predictions:")
        for i, example in enumerate(results['examples'][:5]):
            print(f"\nExample {i+1}:")
            print(f"Content: {example['content_preview']}...")
            for field in self.models.keys():
                print(f"\n{field}:")
                print(f"  Expected: {example['original'].get(field, 'N/A')}")
                print(f"  Predicted: {example['predicted'][field]['prediction']}")
                print(f"  Confidence: {example['predicted'][field]['confidence']:.2f}")

def main():
    """Main evaluation function."""
    try:
        print("\n=== Starting HDL Model Evaluation ===")
        evaluator = HDLEvaluator()
        
        while True:
            print("\nEvaluation Options:")
            print("1. Evaluate on batch from MongoDB")
            print("2. Analyze specific HDL code")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ")
            
            if choice == '1':
                sample_size = int(input("Enter sample size (default 50): ") or 50)
                results = evaluator.evaluate_batch(sample_size)
                evaluator.print_evaluation_report(results)
                
            elif choice == '2':
                print("\nEnter HDL code (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == "":
                        break
                    lines.append(line)
                
                hdl_code = "\n".join(lines)
                if hdl_code.strip():
                    result = evaluator.analyze_single(hdl_code)
                    print("\nAnalysis Result:")
                    for field, data in result.items():
                        print(f"{field}:")
                        print(f"  Prediction: {data['prediction']}")
                        print(f"  Confidence: {data['confidence']:.2f}")
            
            elif choice == '3':
                break
                
    except Exception as e:
        print(f"\nError: {str(e)}")
    finally:
        print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
