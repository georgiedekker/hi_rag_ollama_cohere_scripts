#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import cohere
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Import our text sanitizer module
try:
    from text_sanitizer import sanitize_for_json, prepare_chunk_for_api, safe_json_loads
except ImportError:
    # Try relative import if direct import fails
    from hi_rag.text_sanitizer import sanitize_for_json, prepare_chunk_for_api, safe_json_loads

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def extract_hierarchical_entities(input_file: str, output_file: str):
    """Extract entities from a given file using Cohere AI"""
    logger.info(f"Extracting entities from: {input_file}")
    
    # Get API key from environment
    api_key = os.environ.get("COHERE_API_KEY", "")
    if not api_key:
        logger.error("COHERE_API_KEY environment variable not set")
        return False
    
    # Load input data
    try:
        with open(input_file, 'r') as f:
            text = f.read()
            
        # Create Cohere client
        co = cohere.Client(api_key=api_key)
            
        # Process in chunks to avoid exceeding context length
        chunks = split_text_into_chunks(text, chunk_size=3000)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Process each chunk to extract entities
        all_entities = []
        all_relations = []
        
        # Process chunks in batches to avoid rate limiting
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Sanitize the chunk before creating the prompt
            sanitized_chunk = sanitize_for_json(chunk)
            
            # Create a prompt for entity extraction
            prompt = f"""
            Extract all entities and their relationships from the following text. 
            Return only JSON with the following format:
            {{
                "entities": [
                    {{"id": "1", "name": "Entity Name", "type": "Person/Organization/Location/Concept/etc"}}
                ],
                "relationships": [
                    {{"source": "1", "target": "2", "type": "related_to/part_of/etc"}}
                ]
            }}
            
            TEXT:
            {sanitized_chunk}
            """
            
            # Run the entity extraction
            try:
                response = co.chat(
                    message=prompt,
                    model=os.environ.get("COHERE_CHAT_MODEL", "command"),
                    temperature=0.3,
                    max_tokens=2000
                )
                
                # Parse the response text to get entities and relationships
                try:
                    # Find the JSON part in the response
                    response_text = response.text
                    start_index = response_text.find('{')
                    end_index = response_text.rfind('}') + 1
                    
                    if start_index >= 0 and end_index > start_index:
                        json_text = response_text[start_index:end_index]
                        
                        # Use our safe JSON parser
                        try:
                            result = safe_json_loads(json_text)
                        except ValueError as e:
                            logger.warning(f"Failed to parse JSON safely: {e}")
                            # Fallback to regular parsing as last resort
                            result = json.loads(json_text)
                        
                        # Sanitize the entities and relationships before adding them
                        if "entities" in result:
                            sanitized_entities = [prepare_chunk_for_api(entity) for entity in result["entities"]]
                            all_entities.extend(sanitized_entities)
                        if "relationships" in result:
                            sanitized_relations = [prepare_chunk_for_api(rel) for rel in result["relationships"]]
                            all_relations.extend(sanitized_relations)
                    else:
                        logger.warning(f"No valid JSON found in response for chunk {i+1}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON from response for chunk {i+1}: {e}")
                    # Try to recover the response by sanitizing it
                    try:
                        sanitized_response = sanitize_for_json(response_text)
                        # Try to extract JSON again
                        start_index = sanitized_response.find('{')
                        end_index = sanitized_response.rfind('}') + 1
                        
                        if start_index >= 0 and end_index > start_index:
                            json_text = sanitized_response[start_index:end_index]
                            result = safe_json_loads(json_text)
                            
                            # Add the recovered entities and relationships
                            if "entities" in result:
                                all_entities.extend(result["entities"])
                            if "relationships" in result:
                                all_relations.extend(result["relationships"])
                            logger.info(f"Successfully recovered JSON data from chunk {i+1} after sanitization")
                    except Exception as recover_error:
                        logger.error(f"Failed to recover JSON from chunk {i+1}: {recover_error}")
                
            except Exception as e:
                logger.error(f"Error calling Cohere API for chunk {i+1}: {e}")
        
        # Save the extracted entities and relationships
        result = {
            "entities": all_entities,
            "relationships": all_relations
        }
        
        # Write the output file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Extracted {len(all_entities)} entities and {len(all_relations)} relationships, saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error during entity extraction: {e}")
        return False

def split_text_into_chunks(text: str, chunk_size: int = 3000, overlap: int = 300) -> List[str]:
    """Split text into chunks with optional overlap"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a nice breakpoint (newline or period)
        if end < len(text):
            # Look for a newline
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 1
            else:
                # Look for a period or other sentence ending
                for char in ['.', '!', '?']:
                    period_pos = text.rfind(char, start, end)
                    if period_pos > start + chunk_size // 2:
                        end = period_pos + 1
                        break
        
        # Add the chunk
        chunks.append(text[start:end])
        
        # Move start position for next chunk, with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

async def main():
    """Main function"""
    # Get file paths from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Extract entities using Cohere AI")
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--model", "-m", default="command", help="Cohere model to use")
    args = parser.parse_args()
    
    # Set the model in environment variables
    os.environ["COHERE_CHAT_MODEL"] = args.model
    
    # Make sure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract entities
    result = await extract_hierarchical_entities(args.input, args.output)
    if result:
        logger.info("Entity extraction completed successfully")
    else:
        logger.error("Entity extraction failed")

if __name__ == "__main__":
    asyncio.run(main()) 