import json
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_dataset_to_mongodb():
    # Define file path
    file_path = r"C:\Users\jackf\Documents\GitHub\back-endAI\label_data\ai-hdlcoder-dataset_part000000000000.json"
    
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
        documents = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    document = json.loads(line.strip())
                    documents.append(document)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue

        if documents:
            # Insert all documents in bulk
            result = collection.insert_many(documents)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
        else:
            print("No valid documents found to insert")

        print("Dataset upload completed successfully")

    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    upload_dataset_to_mongodb()