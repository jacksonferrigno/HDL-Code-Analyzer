from pymongo import MongoClient
import os
from dotenv import load_dotenv
from openai import OpenAI
import re
import json
from time import sleep
from datetime import datetime
from tqdm import tqdm
import logging

load_dotenv()

# Add global constants at the top level, after imports
MAX_TOKENS = 12000
MIN_TOKENS = 1000  # Minimum tokens to try before failing
MAX_CONTENT_LENGTH = MAX_TOKENS * 4
def setup_logger():
    """Set up logging with both file and console output"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'hdl_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# Set up OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_technical_details(content):
    """Extract technical details from HDL code"""
    details = {
        'ports': [],
        'signals': [],
        'generics': [],
        'processes': 0,
        'components': [],
        'state_variables': [],
        'clock_domains': set(),
        'reset_signals': set(),
    }
    
    try:
        # Extract ports
        port_matches = re.finditer(r'port\s*\((.*?)\)\s*;', content, re.DOTALL | re.IGNORECASE)
        for match in port_matches:
            port_list = match.group(1).split(';')
            for port in port_list:
                if ':' in port:
                    port = port.strip()
                    if port:
                        details['ports'].append(port)

        # Extract signals
        signal_matches = re.finditer(r'signal\s+(\w+\s*:.*?);', content, re.DOTALL | re.IGNORECASE)
        for match in signal_matches:
            details['signals'].append(match.group(1).strip())

        # Extract generics
        generic_matches = re.finditer(r'generic\s*\((.*?)\)\s*;', content, re.DOTALL | re.IGNORECASE)
        for match in generic_matches:
            generic_list = match.group(1).split(';')
            for generic in generic_list:
                if ':' in generic:
                    generic = generic.strip()
                    if generic:
                        details['generics'].append(generic)

        # Count processes
        details['processes'] = len(re.findall(r'\bprocess\b', content, re.IGNORECASE))

        # Find components
        component_matches = re.finditer(r'component\s+(\w+)\s+is', content, re.IGNORECASE)
        for match in component_matches:
            details['components'].append(match.group(1))

        # Find potential state variables
        state_matches = re.finditer(r'type\s+(\w+)\s+is\s*\((.*?)\)', content, re.DOTALL | re.IGNORECASE)
        for match in state_matches:
            details['state_variables'].append(f"{match.group(1)}: {match.group(2)}")

        # Find clock domains
        clock_matches = re.finditer(r'(rising|falling)_edge\s*\((\w+)\)', content, re.IGNORECASE)
        for match in clock_matches:
            details['clock_domains'].add(match.group(2))

        # Find reset signals
        reset_matches = re.finditer(r'\b(rst|reset|clr|clear)\w*\s*:', content, re.IGNORECASE)
        for match in reset_matches:
            details['reset_signals'].add(match.group(1))

    except Exception as e:
        logger.error(f"Error extracting technical details: {str(e)}")

    # Convert sets to lists for JSON serialization
    details['clock_domains'] = list(details['clock_domains'])
    details['reset_signals'] = list(details['reset_signals'])
    
    return details

def truncate_content(content, max_tokens=12000):
    """Truncate content to stay within token limits while preserving important parts"""
    # Split content into sections
    sections = re.split(r'(entity|architecture|package)\s+', content, flags=re.IGNORECASE)
    
    # If no sections found, do basic truncation
    if len(sections) <= 1:
        return content[:max_tokens * 4]  # Rough estimate of 4 chars per token
        
    # Rebuild content with essential parts
    truncated = ""
    token_estimate = 0
    
    # Always include library declarations
    lib_match = re.search(r'library.*?;(\s*use.*?;)*', content, re.IGNORECASE | re.DOTALL)
    if lib_match:
        truncated += lib_match.group(0) + "\n\n"
        token_estimate += len(lib_match.group(0)) // 4
        
    for i in range(1, len(sections), 2):
        section_type = sections[i].lower()
        section_content = sections[i + 1] if i + 1 < len(sections) else ""
        
        # Extract just the declaration and port/generic sections for entities
        if section_type == 'entity':
            entity_parts = re.split(r'(port|generic)\s+', section_content, flags=re.IGNORECASE)
            if len(entity_parts) > 1:
                truncated += f"entity {entity_parts[0]}"
                # Add port and generic sections if they exist
                for j in range(1, len(entity_parts), 2):
                    section = entity_parts[j].lower()
                    content = entity_parts[j + 1] if j + 1 < len(entity_parts) else ""
                    if section in ('port', 'generic'):
                        truncated += f"{section} {content}"
                        if "end entity" not in content.lower():
                            truncated += "\nend entity;\n\n"
        
        # For architecture, include declaration and key processes
        elif section_type == 'architecture':
            # Get the declaration
            arch_parts = section_content.split('begin', 1)
            if len(arch_parts) > 0:
                truncated += f"architecture {arch_parts[0]}\nbegin\n"
                
                # Add key processes if they exist
                if len(arch_parts) > 1:
                    processes = re.findall(r'process\b.*?end\s+process', arch_parts[1], 
                                        re.IGNORECASE | re.DOTALL)
                    for process in processes[:2]:  # Include up to 2 processes
                        truncated += f"\n{process}\n"
                        
                truncated += "\nend architecture;\n\n"
        
        # For packages, include declaration and type definitions
        elif section_type == 'package':
            pkg_parts = section_content.split('begin', 1)
            if len(pkg_parts) > 0:
                truncated += f"package {pkg_parts[0]}"
                if len(pkg_parts) > 1:
                    truncated += f"begin{pkg_parts[1][:1000]}"  # Limit package body
                if "end package" not in truncated.lower():
                    truncated += "\nend package;\n\n"
        
        # Check estimated token count
        token_estimate = len(truncated) // 4
        if token_estimate >= max_tokens:
            break
    
    return truncated

def analyze_hdl_code(doc, retry_count=3):
    """Analyze HDL code using OpenAI API with retries"""
    content = doc.get('content', '')
    current_tokens = MAX_TOKENS
    
    if not content.strip():
        logger.warning(f"Empty content for document {doc['_id']}")
        return None

    # Initial truncation
    truncated_content = truncate_content(content, current_tokens)
    
    for attempt in range(retry_count):
        try:
            # Extract technical details first
            tech_details = extract_technical_details(truncated_content)
            
            # Create a shorter prompt for large files
            if len(truncated_content) > 5000:
                system_prompt = """Analyze this HDL code and provide a JSON with:
                - component_type: [StateMachine/Counter/ALU/Package/Interface/Memory/Controller/Decoder/TestBench/Other]
                - design_pattern: main pattern used
                - complexity_level: 1-5
                - primary_function: main purpose
                - interface_types: key interfaces
                - key_features: main features
                - ideal_prompt: prompt to generate similar code"""
            else:
                system_prompt = """You are an expert HDL code analyzer and prompt engineer. 
                Analyze the HDL code and provide detailed information about its characteristics and how to generate it.
                Your response should be in JSON format with the following fields:
                - component_type: one of [StateMachine, Counter, ALU, Package, Interface, Memory, Controller, Decoder, TestBench, Other]
                - design_pattern: main design pattern used
                - complexity_level: number from 1-5
                - primary_function: detailed description of main purpose
                - interface_types: array of interface types used
                - key_features: array of technical features
                - implementation_details: array of important implementation choices
                - design_considerations: array of important design decisions and trade-offs
                - performance_characteristics: any timing or performance related aspects
                - potential_applications: array of possible use cases
                - ideal_prompt: a detailed prompt that could be used to generate this kind of HDL code
                - prompt_keywords: array of important keywords that should be in the prompt
                - design_constraints: any constraints that should be specified in the prompt
                - suggested_modifications: potential improvements or variations"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"""Analyze this HDL code and its technical details:

Code:
{truncated_content}

Technical Details:
{json.dumps(tech_details, indent=2)}"""}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            result_text = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if not json_match:    
                raise ValueError("No JSON found in response")
                
            result = json.loads(json_match.group())
            
            # Add technical details and metadata
            result['technical_details'] = tech_details
            result['analysis_timestamp'] = datetime.utcnow().isoformat()
            result['truncated'] = len(truncated_content) < len(content)
            result['original_length'] = len(content)
            result['analyzed_length'] = len(truncated_content)
            result['truncation_ratio'] = len(truncated_content) / len(content)
            
            return result
            
        except Exception as e:
            if 'context_length_exceeded' in str(e):
                logger.warning(f"Context length exceeded at {current_tokens} tokens, reducing size...")
                current_tokens = current_tokens // 2
                
                # Check if we've hit minimum token limit
                if current_tokens < MIN_TOKENS:
                    logger.warning(f"Document {doc['_id']} too large even at minimum size")
                    return None
                    
                # Try more aggressive truncation
                truncated_content = truncate_content(truncated_content, current_tokens)
                continue
                
            elif 'Rate limit' in str(e):
                wait_time = (attempt + 1) * 30
                logger.warning(f"Rate limit hit, waiting {wait_time} seconds...")
                sleep(wait_time)
                continue
                
            elif attempt < retry_count - 1:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                sleep(5)
                continue
                
            else:
                logger.error(f"All attempts failed for document {doc['_id']}: {str(e)}")
                return None


def process_documents(collection, target_count=250, batch_size=20):
    """Process specified number of documents with cursor timeout handling and analysis check"""
    processed_count = 0
    success_count = 0
    skipped_count = 0
    last_id = None
    
    while processed_count < target_count:
        try:
            # Query with pagination using _id and check for existing analysis
            query = {
                "$or": [
                    {"analysis": {"$exists": False}},
                    {"analysis": None}
                ],
                "processing_failed": {"$ne": True}
            }
            if last_id:
                query["_id"] = {"$gt": last_id}
                
            # Get current count of remaining documents
            remaining_docs = collection.count_documents(query)
            if remaining_docs == 0:
                logger.info("No more unanalyzed documents to process")
                break
                
            # Get a batch of documents
            cursor = collection.find(
                query,
                {"_id": 1, "content": 1, "analysis": 1}
            ).limit(batch_size)
            
            # Convert cursor to list immediately to avoid timeout
            batch_docs = list(cursor)
            
            if not batch_docs:
                logger.info("No more documents to process")
                break
                
            # Process the batch
            for doc in batch_docs:
                try:
                    doc_id = doc['_id']
                    last_id = doc_id  # Update last_id for pagination
                    
                    # Double check if document already has analysis
                    if 'analysis' in doc and doc['analysis'] is not None:
                        logger.info(f"Skipping already analyzed document {doc_id}")
                        skipped_count += 1
                        processed_count += 1
                        continue
                    
                    logger.info(f"Processing document {doc_id}")
                    
                    analysis = analyze_hdl_code(doc)
                    
                    if analysis:
                        # Final check before update to prevent duplicate analysis
                        existing = collection.find_one(
                            {"_id": doc_id, "analysis": {"$exists": True}},
                            {"_id": 1}
                        )
                        
                        if existing:
                            logger.info(f"Document {doc_id} was analyzed by another process")
                            skipped_count += 1
                        else:
                            collection.update_one(
                                {"_id": doc_id},
                                {
                                    "$set": {
                                        "analysis": analysis,
                                        "analysis_timestamp": datetime.utcnow()
                                    }
                                }
                            )
                            success_count += 1
                            logger.info(f"Successfully analyzed document {doc_id}")
                    else:
                        logger.warning(f"Failed to analyze document {doc_id}")
                        collection.update_one(
                            {"_id": doc_id},
                            {"$set": {"processing_failed": True}}
                        )
                    
                    processed_count += 1
                    
                    # Save progress summary every 10 documents
                    if processed_count % 10 == 0:
                        progress_summary = {
                            "processed": processed_count,
                            "successful": success_count,
                            "skipped": skipped_count,
                            "last_id": str(last_id),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        with open('analysis_progress.json', 'w') as f:
                            json.dump(progress_summary, f, indent=2)
                        
                        logger.info(f"Progress: {processed_count} processed, {success_count} successful, {skipped_count} skipped")
                    
                    # Rate limiting pause every 50 requests
                    if processed_count % 50 == 0:
                        logger.info("Taking a short break to avoid rate limits...")
                        sleep(10)
                    
                    # Check if we've hit the target
                    if processed_count >= target_count:
                        break
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {str(e)}")
                    continue
            
        except Exception as e:
            logger.error(f"Batch processing error: {str(e)}")
            # Wait before retrying
            sleep(5)
            continue
    
    logger.info(f"\nProcessing complete.")
    logger.info(f"Processed: {processed_count}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Skipped: {skipped_count}")
    return success_count
def main():
    try:
        logger.info("Connecting to MongoDB...")
        client = MongoClient(os.getenv('DB_URI'))
        db = client['hdl_database']
        collection = db['hdl_codes']
        
        # Get current counts
        total_docs = collection.count_documents({})
        analyzed_docs = collection.count_documents({"analysis": {"$exists": True}})
        failed_docs = collection.count_documents({"processing_failed": True})
        remaining_docs = collection.count_documents({
            "$or": [
                {"analysis": {"$exists": False}},
                {"analysis": None}
            ],
            "processing_failed": {"$ne": True}
        })
        
        logger.info(f"Database status:")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Already analyzed: {analyzed_docs}")
        logger.info(f"Failed documents: {failed_docs}")
        logger.info(f"Remaining to analyze: {remaining_docs}")
        
        if remaining_docs == 0:
            logger.info("No documents remaining to analyze.")
            return
        
        # Check for existing progress
        try:
            with open('analysis_progress.json', 'r') as f:
                progress = json.load(f)
                processed = progress.get('processed', 0)
                remaining = 250 - processed
                if remaining > 0:
                    logger.info(f"Resuming from previous run. {remaining} documents remaining.")
                    successful = process_documents(collection, target_count=remaining)
                else:
                    logger.info("Previous run completed all documents.")
                    return
        except FileNotFoundError:
            logger.info("Starting new processing run...")
            successful = process_documents(collection, target_count=250)
        
        logger.info(f"Analysis complete. Successfully analyzed {successful} documents.")
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")
if __name__ == "__main__":
    main()