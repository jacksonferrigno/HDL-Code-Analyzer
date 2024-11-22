from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import json
from time import sleep
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List

class HDLAnalysis:
    """Handles HDL code analysis using OpenAI API with MongoDB integration."""
    
    MAX_TOKENS = 16000  # Setting slightly below max to be safe
    MIN_TOKENS = 1000   # Minimum tokens to maintain meaningful analysis
    
    def __init__(self, batch_size: int = 20, target_count: int = 250):
        """Initialize connections and configurations for the HDL analysis.

        Args:
            batch_size (int): Number of documents to process in each batch.
            target_count (int): Total number of documents to analyze.
        """
        load_dotenv()  # Load environment variables from .env file
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))  # Initialize OpenAI client
        self.mongo_client = MongoClient(os.getenv('DB_URI'))  # Connect to MongoDB
        self.db = self.mongo_client['hdl_database']  # Select the database
        self.collection = self.db['hdl_codes']  # Select the collection
        self.logger = self._setup_logger()  # Set up logging
        self.batch_size = batch_size  # Set batch size for processing
        self.target_count = target_count  # Set target count for documents to analyze
        self.stats = {
            'total_attempts': 0,
            'successful_analyses': 0,
            'failed_analyses': 0,
            'skipped_documents': 0,
            'processed_count': 0
        }

    def _setup_logger(self) -> logging.Logger:
        """Configure the logging system to log messages to both console and file.

        Returns:
            logging.Logger: Configured logger instance.
        """
        log_filename = f'hdl_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        
        # Set up file handler for logging to a file
        file_handler = logging.FileHandler(log_filename)
        file_handler.setFormatter(formatter)
        
        # Set up console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)  # Set logging level to INFO
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        logger.info("Starting new HDL analysis session")  # Log session start
        return logger

    def _get_system_prompt(self, content_length: int) -> str:
        """Select appropriate prompt based on the length of the HDL code.

        Args:
            content_length (int): Length of the HDL code content.

        Returns:
            str: The system prompt to be used for the OpenAI API.
        """
        if content_length > 5000:
            return """Analyze this HDL code and provide a JSON with:
                - component_type: [StateMachine/Counter/ALU/Package/Interface/Memory/Controller/Decoder/TestBench/Other]
                - design_pattern: main pattern used
                - complexity_level: 1-5
                - primary_function: main purpose
                - interface_types: key interfaces
                - key_features: main features
                - ideal_prompt: prompt to generate similar code"""
        else:
            return """You are an expert HDL code analyzer. 
                Analyze the HDL code and provide detailed information in JSON format with fields:
                - component_type, design_pattern, complexity_level, primary_function
                - interface_types, key_features, implementation_details
                - design_considerations, performance_characteristics
                - potential_applications, ideal_prompt, prompt_keywords
                - design_constraints, suggested_modifications"""

    def _truncate_hdl(self, hdl_code: str) -> str:
        """Intelligently truncate HDL code to fit token limits.

        Args:
            hdl_code (str): The HDL code to be truncated.

        Returns:
            str: The truncated HDL code.
        """
        sections = re.split(r'(entity|architecture|package)\s+', hdl_code, flags=re.IGNORECASE)
        
        # If no sections found, return a portion of the code
        if len(sections) <= 1:
            return hdl_code[:self.MAX_TOKENS * 4]
            
        truncated = ""
        
        # Preserve library declarations
        lib_match = re.search(r'library.*?;(\s*use.*?;)*', hdl_code, re.IGNORECASE | re.DOTALL)
        if lib_match:
            truncated += lib_match.group(0) + "\n\n"
        
        # Iterate through sections to build the truncated code
        for i in range(1, len(sections), 2):
            section_type = sections[i].lower()
            section_content = sections[i + 1] if i + 1 < len(sections) else ""
            
            if section_type == 'entity':
                entity_parts = re.split(r'(port|generic)\s+', section_content, flags=re.IGNORECASE)
                if len(entity_parts) > 1:
                    truncated += f"entity {entity_parts[0]}"
                    for j in range(1, len(entity_parts), 2):
                        if j + 1 < len(entity_parts) and entity_parts[j].lower() in ('port', 'generic'):
                            truncated += f"{entity_parts[j]} {entity_parts[j+1]}"
                            if "end entity" not in entity_parts[j+1].lower():
                                truncated += "\nend entity;\n\n"
            
            elif section_type == 'architecture':
                arch_parts = section_content.split('begin', 1)
                if len(arch_parts) > 0:
                    truncated += f"architecture {arch_parts[0]}\nbegin\n"
                    if len(arch_parts) > 1:
                        processes = re.findall(r'process\b.*?end\s+process', 
                                            arch_parts[1], re.IGNORECASE | re.DOTALL)
                        truncated += "\n".join(processes[:3])  # Include first 3 processes
                    truncated += "\nend architecture;\n\n"
            
            elif section_type == 'package':
                pkg_parts = section_content.split('begin', 1)
                if len(pkg_parts) > 0:
                    truncated += f"package {pkg_parts[0]}"
                    if "end package" not in truncated.lower():
                        truncated += "\nend package;\n\n"
            
            # Stop if the truncated content exceeds the maximum token limit
            if len(truncated) > self.MAX_TOKENS * 3:
                break
        
        return truncated

    def get_unanalyzed_documents(self, last_id: Optional[str] = None) -> List[Dict]:
        """Retrieve a batch of unanalyzed documents from MongoDB.

        Args:
            last_id (Optional[str]): The last document ID processed for pagination.

        Returns:
            List[Dict]: A list of documents that have not been analyzed.
        """
        query = {
            "$or": [
                {"analysis": {"$exists": False}},
                {"analysis": None}
            ],
            "processing_failed": {"$ne": True}
        }
        if last_id:
            query["_id"] = {"$gt": last_id}

        return list(self.collection.find(
            query,
            {"_id": 1, "content": 1}
        ).limit(self.batch_size))

    def analyze_hdl(self, hdl_code: str, doc_id: str, retry_count: int = 3) -> Optional[Dict[str, Any]]:
        """Analyze HDL code using OpenAI API with token handling.

        Args:
            hdl_code (str): The HDL code to analyze.
            doc_id (str): The document ID for logging purposes.
            retry_count (int): Number of attempts to analyze the code.

        Returns:
            Optional[Dict[str, Any]]: The analysis result as a dictionary or None if failed.
        """
        if not hdl_code.strip():
            self.logger.error(f"Empty HDL code for document {doc_id}")
            return None

        self.stats['total_attempts'] += 1
        original_length = len(hdl_code)
        current_content = hdl_code
        
        for attempt in range(retry_count):
            try:
                # Handle large content by truncating if necessary
                if len(current_content) > self.MAX_TOKENS * 4:
                    self.logger.info(f"Content too large ({len(current_content)} chars), truncating...")
                    current_content = self._truncate_hdl(current_content)
                    self.logger.info(f"Truncated to {len(current_content)} chars")

                system_prompt = self._get_system_prompt(len(current_content))
                
                # Send the request to OpenAI API
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this HDL code:\n\n{current_content}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )

                # Extract the JSON result from the response
                result = json.loads(re.search(r'\{.*\}', response.choices[0].message.content, re.DOTALL).group())
                result.update({
                    'analysis_timestamp': datetime.utcnow().isoformat(),
                    'content_length': len(current_content),
                    'original_length': original_length,
                    'was_truncated': len(current_content) < original_length
                })

                self.stats['successful_analyses'] += 1
                return result

            except Exception as e:
                # Handle specific exceptions and retry logic
                if 'context_length_exceeded' in str(e):
                    if attempt < retry_count - 1:
                        current_content = self._truncate_hdl(current_content)
                        if len(current_content) < self.MIN_TOKENS:
                            self.logger.error(f"Content too short after truncation for {doc_id}")
                            break
                        continue

                if 'Rate limit' in str(e):
                    wait_time = (attempt + 1) * 30
                    self.logger.warning(f"Rate limit hit for {doc_id}, waiting {wait_time}s...")
                    sleep(wait_time)
                    continue
                
                if attempt < retry_count - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {doc_id}: {str(e)}")
                    sleep(5)
                    continue
                
                self.logger.error(f"Analysis failed for {doc_id} after {retry_count} attempts: {str(e)}")
                self.stats['failed_analyses'] += 1
                return None

    def process_batch(self) -> None:
        """Process a batch of documents from MongoDB.

        This method retrieves documents that have not been analyzed and processes them
        by calling the analyze_hdl method. It also handles logging and progress tracking.
        """
        last_id = None
        
        while self.stats['processed_count'] < self.target_count:
            try:
                documents = self.get_unanalyzed_documents(last_id)
                if not documents:
                    self.logger.info("No more documents to process")
                    break

                for doc in documents:
                    doc_id = doc['_id']
                    last_id = doc_id

                    # Check if the document has already been analyzed
                    if self.collection.find_one({"_id": doc_id, "analysis": {"$exists": True}}):
                        self.logger.info(f"Document {doc_id} already analyzed")
                        self.stats['skipped_documents'] += 1
                        continue

                    self.logger.info(f"Processing document {doc_id}")
                    analysis = self.analyze_hdl(doc.get('content', ''), str(doc_id))

                    if analysis:
                        # Update the document with the analysis result
                        self.collection.update_one(
                            {"_id": doc_id},
                            {
                                "$set": {
                                    "analysis": analysis,
                                    "analysis_timestamp": datetime.utcnow()
                                }
                            }
                        )
                        self.logger.info(f"Successfully analyzed document {doc_id}")
                    else:
                        # Mark the document as failed if analysis was not successful
                        self.collection.update_one(
                            {"_id": doc_id},
                            {"$set": {"processing_failed": True}}
                        )
                        self.logger.warning(f"Failed to analyze document {doc_id}")

                    self.stats['processed_count'] += 1
                    
                    # Save progress every 10 documents
                    if self.stats['processed_count'] % 10 == 0:
                        self._save_progress(last_id)
                        self.log_stats()

                    # Rate limiting: take a break every 50 documents
                    if self.stats['processed_count'] % 50 == 0:
                        self.logger.info("Taking a break to avoid rate limits...")
                        sleep(10)

                    if self.stats['processed_count'] >= self.target_count:
                        break

            except Exception as e:
                self.logger.error(f"Batch processing error: {str(e)}")
                sleep(5)

    def _save_progress(self, last_id: str) -> None:
        """Save current progress to a JSON file.

        Args:
            last_id (str): The last document ID processed.
        """
        progress = {
            **self.stats,
            "last_id": str(last_id),
            "timestamp": datetime.utcnow().isoformat()
        }
        with open('analysis_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

    def log_stats(self) -> None:
        """Log current statistics of the analysis process."""
        self.logger.info("Current Statistics:")
        for key, value in self.stats.items():
            self.logger.info(f"{key.replace('_', ' ').title()}: {value}")

    def run(self) -> None:
        """Main execution method to start the analysis process.

        This method loads existing progress, processes batches of documents,
        and logs the results.
        """
        try:
            # Load existing progress from file
            try:
                with open('analysis_progress.json', 'r') as f:
                    progress = json.load(f)
                    self.stats['processed_count'] = progress.get('processed_count', 0)
                    remaining = self.target_count - self.stats['processed_count']
                    if remaining > 0:
                        self.logger.info(f"Resuming from previous run. {remaining} documents remaining.")
                        self.target_count = remaining
                    else:
                        self.logger.info("Previous run completed.")
                        return
            except FileNotFoundError:
                self.logger.info("Starting new processing run...")

            self.process_batch()  # Start processing documents
            self.log_stats()  # Log final statistics
            
        except Exception as e:
            self.logger.error(f"Fatal error: {str(e)}")
        finally:
            self.close()  # Ensure resources are cleaned up

    def close(self) -> None:
        """Cleanup and close connections to MongoDB."""
        if self.mongo_client:
            self.mongo_client.close()
            self.logger.info("MongoDB connection closed")
        self.logger.info("HDL Analysis session completed")

def main():
    """Main execution function for HDL Analysis.

    This function initializes the HDLAnalysis class and starts the analysis process.
    It also prints the initial status of the MongoDB database.
    """
    try:
        print("\n=== Starting HDL Code Analysis System ===\n")
        
        # Initialize the analyzer with specified batch size and target count
        analyzer = HDLAnalysis(
            batch_size=20,     # Process 20 documents at a time
            target_count=250   # Process a total of 250 documents
        )
        
        # Print initial MongoDB status
        total_docs = analyzer.collection.count_documents({})
        analyzed_docs = analyzer.collection.count_documents({"analysis": {"$exists": True}})
        failed_docs = analyzer.collection.count_documents({"processing_failed": True})
        remaining_docs = analyzer.collection.count_documents({
            "$or": [
                {"analysis": {"$exists": False}},
                {"analysis": None}
            ],
            "processing_failed": {"$ne": True}
        })
        
        print(f"Database Status:")
        print(f"Total documents: {total_docs}")
        print(f"Already analyzed: {analyzed_docs}")
        print(f"Failed documents: {failed_docs}")
        print(f"Remaining to analyze: {remaining_docs}\n")
        
        if remaining_docs == 0:
            print("No documents remaining to analyze.")
            return
            
        # Run the analysis
        print("Starting analysis process...")
        analyzer.run()
        
    except Exception as e:
        print(f"\nFatal error occurred: {str(e)}")
    finally:
        print("\n=== HDL Code Analysis Complete ===\n")

if __name__ == "__main__":
    main()