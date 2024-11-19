from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import json
from time import sleep
load_dotenv()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def clean_hdl_code(content):
    """Remove comments and extract core HDL code"""
    # Remove single-line comments
    content = re.sub(r'--.*$', '', content, flags=re.MULTILINE)
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Get essential parts (entity, architecture, package)
    core_parts = []
    
    # Extract entity
    entity_match = re.search(r'entity\s+\w+\s+is.*?end\s+\w+;', content, re.DOTALL | re.IGNORECASE)
    if entity_match:
        core_parts.append(entity_match.group())
    
    # Extract architecture
    arch_match = re.search(r'architecture\s+\w+\s+of\s+\w+\s+is.*?begin.*?end\s+\w+;', content, re.DOTALL | re.IGNORECASE)
    if arch_match:
        core_parts.append(arch_match.group())
    
    # Extract package
    package_match = re.search(r'package\s+\w+\s+is.*?end\s+\w+;', content, re.DOTALL | re.IGNORECASE)
    if package_match:
        core_parts.append(package_match.group())
    
    cleaned = '\n'.join(core_parts) if core_parts else content
    return ' '.join(cleaned.split())  # Normalize whitespace

def analyze_hdl_code(doc):
    """Analyze HDL code using OpenAI API"""
    content = doc.get('content', '')
    
    if not content.strip():
        print(f"Empty content for document {doc['_id']}")
        return None
    
    # Clean and extract core HDL code
    cleaned_code = clean_hdl_code(content)
    if not cleaned_code:
        print(f"No valid HDL code found in document {doc['_id']}")
        return None

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """You are an expert HDL code analyzer. 
                Analyze HDL code and provide specific information about its characteristics.
                Your response should be in JSON format with specific fields:
                - component_type: one of [StateMachine, Counter, ALU, Package, Interface, Memory, Controller, Decoder, TestBench, Other]
                - design_pattern: main design pattern in 2-4 words
                - complexity_level: number from 1-5
                - primary_function: main purpose in 5-10 words
                - interface_types: array of 2-3 interface types
                - key_features: array of 2-3 technical features"""},
                {"role": "user", "content": f"Analyze this HDL code:\n{cleaned_code}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        # Get the response text
        result_text = response.choices[0].message.content
        
        # Try to parse JSON from the response
        try:
            # Look for JSON in the response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Validate and clean up the result
            result = {
                "component_type": result.get("component_type", "other").lower(),
                "design_pattern": result.get("design_pattern", "digital logic"),
                "complexity_level": min(max(1, int(result.get("complexity_level", 3))), 5),
                "primary_function": result.get("primary_function", "digital logic processing"),
                "interface_types": result.get("interface_types", ["clock", "data"])[:3],
                "key_features": result.get("key_features", ["synchronous"])[:3]
            }
            
            # Add metadata
            result['original_repo'] = doc.get('repo_name', '')
            result['original_path'] = doc.get('path', '')
            result['original_size'] = doc.get('size', '')
            result['original_license'] = doc.get('license', '')
            
            return result
            
        except Exception as e:
            print(f"Error parsing OpenAI response: {str(e)}")
            print("Raw response:", result_text)
            return None
            
    except Exception as e:
        if 'Rate limit' in str(e):
            print("Rate limit hit, waiting 60 seconds...")
            sleep(60)
            return analyze_hdl_code(doc)  # Retry
        else:
            print(f"OpenAI API error: {str(e)}")
            return None

def process_batch(collection, batch_size=10):
    """Process documents in batches"""
    unlabeled_docs = collection.find({
        "labels": {"$exists": False},
        "processing_failed": {"$ne": True}
    }).limit(batch_size)
    
    processed_count = 0
    success_count = 0
    
    for doc in unlabeled_docs:
        try:
            processed_count += 1
            print(f"\nProcessing document {doc['_id']} ({processed_count}/{batch_size})...")
            
            labels = analyze_hdl_code(doc)
            
            if labels:
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"labels": labels}}
                )
                success_count += 1
                print(f"Successfully labeled document {doc['_id']}")
                print("Labels:", json.dumps(labels, indent=2))
            else:
                print(f"Failed to generate labels for document {doc['_id']}")
                collection.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"processing_failed": True}}
                )
                
        except Exception as e:
            print(f"Error processing document {doc['_id']}: {str(e)}")
            continue
    
    print(f"\nBatch processing complete. Processed {processed_count} documents, {success_count} successful.")
    return processed_count > 0

def main():
    try:
        print("Connecting to MongoDB...")
        client = MongoClient(os.getenv('DB_URI'))
        db = client['hdl_database']
        collection = db['hdl_codes']
        
        print("Starting document processing...")
        while True:
            has_more = process_batch(collection)
            
            if not has_more:
                print("No more documents to process.")
                break
                
            response = input("Process another batch? (y/n): ")
            if response.lower() != 'y':
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()