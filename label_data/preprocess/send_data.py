import json
from pymongo import MongoClient
import os
from dotenv import load_dotenv
import re
import logging
from datetime import datetime

# Load environment variables
load_dotenv()

def setup_logger():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'hdl_upload_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def clean_hdl_content(content):
    """Clean HDL code by removing content before library declaration"""
    try:
        # Find the first library declaration
        library_match = re.search(
            r'\b(?:library\s+(?:ieee|IEEE|work|std|STD))',
            content,
            re.IGNORECASE | re.MULTILINE
        )
        
        if library_match:
            return content[library_match.start():].strip()
        
        # Fallback to entity/architecture if no library found
        alt_match = re.search(
            r'\b(?:entity|architecture|package)\s+\w+\s+is',
            content,
            re.IGNORECASE | re.MULTILINE
        )
        
        if alt_match:
            return content[alt_match.start():].strip()
        
        return content.strip()
        
    except Exception as e:
        return None

def upload_dataset_to_mongodb():
    # Define file path
    file_path = r"C:\Users\jackf\Documents\GitHub\back-endAI\label_data\ai-hdlcoder-dataset_part000000000001.json"
    
    logger = setup_logger()
    
    try:
        # Get MongoDB connection string from environment variable
        mongodb_uri = os.getenv('DB_URI')
        
        if not mongodb_uri:
            raise ValueError("DB_URI environment variable not found")

        # Connect to MongoDB
        client = MongoClient(mongodb_uri)
        
        # Create or get database
        db = client['hdl_database']
        
        # Create or get collection
        collection = db['hdl_codes']

        # Read JSONL file line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    document = json.loads(line.strip())
                    content = document.get('content', '')
                    
                    if content:
                        # Clean the content
                        cleaned_content = clean_hdl_content(content)
                        if cleaned_content:
                            # Remove unwanted fields
                            for field in ['repo_name', 'path', 'copies', 'size', 'license', 'labels', 'processing_failed']:
                                document.pop(field, None)  # Remove field if it exists
                            document['content'] = cleaned_content
                            
                            # Insert cleaned document into MongoDB
                            collection.insert_one(document)
                        else:
                            logger.warning(f"Cleaning failed for document: {document}")
                    else:
                        logger.warning("Empty content found in document, skipping.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing line: {e}")
                    continue

        logger.info("Processing complete. All documents have been cleaned and inserted into MongoDB.")

    except FileNotFoundError:
        logger.error(f"Error: Dataset file not found at {file_path}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    upload_dataset_to_mongodb()