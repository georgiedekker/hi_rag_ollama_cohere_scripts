#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${YELLOW}[PIPELINE]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Load environment variables from .env file
load_env_vars() {
    print_status "Looking for .env file"
    if [ -f ".env" ]; then
        print_status "Loading environment variables from .env file"
        set -o allexport
        source .env
        set +o allexport
    elif [ -f "../.env" ]; then
        print_status "Loading environment variables from ../.env file"
        set -o allexport
        source ../.env
        set +o allexport
    else
        print_status "No .env file found in current or parent directory. Ensure NEO4J variables are set if needed."
    fi
}

# Load the environment variables first
load_env_vars

# Check if a Cohere API key is provided
if [ -z "$COHERE_API_KEY" ]; then
    print_error "COHERE_API_KEY environment variable is not set"
    print_status "Please set your Cohere API key in .env file or directly with: export COHERE_API_KEY=your_api_key"
    exit 1
fi

print_status "Found Cohere API key: ${COHERE_API_KEY:0:3}...${COHERE_API_KEY: -3}"

# Set up Cohere configuration
if [ -z "$COHERE_EMBEDDING_MODEL" ]; then
    export COHERE_EMBEDDING_MODEL="embed-english-v3.0"
    print_status "Using default Cohere embedding model: $COHERE_EMBEDDING_MODEL"
else
    print_status "Using Cohere embedding model from environment: $COHERE_EMBEDDING_MODEL"
fi

if [ -z "$COHERE_CHAT_MODEL" ]; then
    export COHERE_CHAT_MODEL="command"
    print_status "Using default Cohere chat model: $COHERE_CHAT_MODEL"
else
    print_status "Using Cohere chat model from environment: $COHERE_CHAT_MODEL"
fi

if [ -z "$COHERE_INPUT_TYPE" ]; then
    export COHERE_INPUT_TYPE="search_document"
    print_status "Using default Cohere input type: $COHERE_INPUT_TYPE"
else
    print_status "Using Cohere input type from environment: $COHERE_INPUT_TYPE"
fi

if [ -z "$COHERE_EMBEDDING_DIM" ]; then
    export COHERE_EMBEDDING_DIM="1024"
    print_status "Using default Cohere embedding dimension: $COHERE_EMBEDDING_DIM"
else
    print_status "Using Cohere embedding dimension from environment: $COHERE_EMBEDDING_DIM"
fi

# Set Cohere as the provider and explicitly unset conflicting environment variables
export PROVIDER="cohere"
print_status "Setting provider to Cohere"

# Explicitly override OpenAI API base to prevent Ollama usage
print_status "Overriding OpenAI API endpoint settings to prevent Ollama usage"
unset OPENAI_API_BASE
unset OPENAI_BASE_URL
unset OLLAMA_BASE_URL

# Check for required text_sanitizer.py file
print_status "Checking for text_sanitizer.py module"
if [ ! -f "text_sanitizer.py" ]; then
    print_status "text_sanitizer.py not found, creating it for safe JSON handling"
    
    # Create the sanitizer module
    cat > text_sanitizer.py << 'EOF'
"""
Text sanitizer module for Cohere API requests.
Ensures all text chunks are properly sanitized with escaped special characters.
"""
import json
import re
import logging
from typing import Union, Dict, Any, List

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sanitize_for_json(text: str) -> str:
    """
    Sanitize text to ensure it can be safely embedded in JSON.
    
    Args:
        text: The text string to sanitize
        
    Returns:
        Sanitized text string safe for JSON inclusion
    """
    if not text:
        return ""
    
    # Replace JSON control characters
    text = text.replace('\\', '\\\\')  # Escape backslashes first
    text = text.replace('"', '\\"')    # Escape double quotes
    text = text.replace('\b', '\\b')   # Escape backspace
    text = text.replace('\f', '\\f')   # Escape form feed
    text = text.replace('\n', '\\n')   # Escape newline
    text = text.replace('\r', '\\r')   # Escape carriage return
    text = text.replace('\t', '\\t')   # Escape tab
    
    # Remove control characters that break JSON
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    return text

def validate_json_safety(obj: Union[Dict, List, str, int, float, bool, None]) -> Union[Dict, List, str, int, float, bool, None]:
    """
    Recursively sanitize all string values in a Python object to ensure JSON safety.
    
    Args:
        obj: The Python object to sanitize (can be dict, list, or primitive)
        
    Returns:
        Sanitized Python object
    """
    if isinstance(obj, dict):
        return {k: validate_json_safety(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [validate_json_safety(item) for item in obj]
    elif isinstance(obj, str):
        return sanitize_for_json(obj)
    else:
        # Return int, float, bool, None as is
        return obj

def prepare_chunk_for_api(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare a chunk for submission to the Cohere API by ensuring all text fields
    are properly sanitized for JSON safety.
    
    Args:
        chunk: Dictionary containing chunk data
        
    Returns:
        Sanitized chunk dictionary
    """
    # Sanitize the entire chunk object
    sanitized_chunk = validate_json_safety(chunk)
    
    # Validate the chunk can be properly serialized
    try:
        json_str = json.dumps(sanitized_chunk)
        # Try parsing it back to ensure it's valid
        json.loads(json_str)
        return sanitized_chunk
    except Exception as e:
        logger.error(f"Error validating chunk after sanitization: {str(e)}")
        # If there's still an issue, use a more aggressive approach
        # Convert to string, sanitize, and convert back
        chunk_str = json.dumps(chunk, ensure_ascii=True)
        sanitized_str = sanitize_for_json(chunk_str)
        
        try:
            return json.loads(sanitized_str)
        except Exception as e2:
            logger.error(f"Failed to repair chunk JSON: {str(e2)}")
            # Return a minimal valid chunk with error message
            return {
                "content": f"[ERROR: Could not sanitize chunk - {str(e2)}]",
                "error": True,
                "original_chunk_length": len(str(chunk))
            }

def safe_json_loads(json_str: str) -> Dict[str, Any]:
    """
    Safely load JSON string, with error handling.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as Python dict/list
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON errors
        logger.warning(f"JSON decode error: {str(e)}")
        
        # Fix for missing commas
        if "Expecting ',' delimiter" in str(e):
            position = e.pos
            # Insert comma at the position
            fixed_json = json_str[:position] + "," + json_str[position:]
            try:
                return json.loads(fixed_json)
            except:
                pass
                
        # If no fixes worked, raise the original error
        raise ValueError(f"Could not parse JSON: {str(e)}")
EOF
    
    print_success "Created text_sanitizer.py with JSON safety functions"
else
    print_status "text_sanitizer.py module found, will use it for JSON safety"
fi

# Create local temporary directory
TEMP_PARENT_DIR="./@temp"
mkdir -p "$TEMP_PARENT_DIR"
if [ $? -ne 0 ]; then
    print_error "Failed to create temporary parent directory: $TEMP_PARENT_DIR"
    exit 1
fi
print_status "Ensured local temporary directory exists: $TEMP_PARENT_DIR"


# Create working directories within the local temp directory
WORK_DIR=$(mktemp -d -p "$TEMP_PARENT_DIR")
if [ $? -ne 0 ] || [ -z "$WORK_DIR" ]; then
    print_error "Failed to create working directory in $TEMP_PARENT_DIR"
    exit 1
fi
print_status "Created working directory: $WORK_DIR"

# Create output directory
OUTPUT_DIR="$WORK_DIR/output"
mkdir -p "$OUTPUT_DIR"
if [ $? -ne 0 ]; then
    print_error "Failed to create output directory: $OUTPUT_DIR"
    rm -rf "$WORK_DIR" # Clean up work dir if output dir creation fails
    exit 1
fi
print_status "Created output directory: $OUTPUT_DIR"


# Set up the file paths
INGEST_DIR="$1"
NER_DIR="$2"
CHUNKER_DIR="$3"

if [ -z "$INGEST_DIR" ] || [ -z "$NER_DIR" ] || [ -z "$CHUNKER_DIR" ]; then
    print_error "Missing required directories. Usage: $0 ingest_dir ner_dir chunker_dir"
    rm -rf "$WORK_DIR" # Clean up created dirs
    exit 1
fi

# Check if input directories exist
if [ ! -d "$INGEST_DIR" ]; then
    print_error "Ingest directory not found: $INGEST_DIR"
    rm -rf "$WORK_DIR"
    exit 1
fi
if [ ! -d "$NER_DIR" ]; then
    print_error "NER directory not found: $NER_DIR"
    rm -rf "$WORK_DIR"
    exit 1
fi
if [ ! -d "$CHUNKER_DIR" ]; then
    print_error "Chunker directory not found: $CHUNKER_DIR"
    rm -rf "$WORK_DIR"
    exit 1
fi

# Process all files in the directories
print_status "Processing all files with Cohere embedding and entity extraction"

# Test Cohere embedding functionality first
print_status "Testing Cohere embedding..."
python -c "
import asyncio
from cohere_embedding import test_embedding

# Run the test
asyncio.run(test_embedding())
print('Cohere embedding test successful!')
"

if [ $? -ne 0 ]; then
    print_error "Cohere embedding test failed. Please check your API key and configuration."
    rm -rf "$WORK_DIR"
    exit 1
fi

print_success "Cohere embedding test successful!"

# Process ingest files
print_status "Processing ingest files from: $INGEST_DIR and loading directly to Neo4j"

# Ensure Neo4j import script exists if we are calling it directly
# Optional: Check for Neo4j credentials here if needed for early exit
# if [ -z "$NEO4J_URL" ] || [ -z "$NEO4J_USER" ] || [ -z "$NEO4J_PASSWORD" ]; then
#     print_error "Neo4j connection details (NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD) not set in environment."
#     # Decide if this is a fatal error for this script's purpose
#     # exit 1 
# fi

find "$INGEST_DIR" -type f -not -name ".*" -not -name ".DS_Store" | while IFS= read -r file; do
    # Skip __init__ files
    if [[ "$(basename "$file")" == "__init__"* ]]; then
        print_status "Skipping __init__ file: $file"
        continue
    fi
    
    print_status "Processing ingest file and loading to Neo4j: $file"
    filename=$(basename "$file")
    # No longer need the intermediate output file path
    # output_file="$OUTPUT_DIR/${filename%.*}_entities.json"
    
    # Run entity extraction AND Neo4j loading
    # IMPORTANT: Assumes mini_entity_extract.py is modified to handle --load-neo4j
    # Alternatively, replace with a different script designed for this.
    python mini_entity_extract.py --input "$file" --load-neo4j
    PROCESS_EXIT_CODE=$?

    if [ $PROCESS_EXIT_CODE -eq 0 ]; then
        print_success "Successfully processed and loaded entities/relationships from $filename to Neo4j"
    else
        print_error "Failed to process/load $filename (Exit code: $PROCESS_EXIT_CODE). Check Python script logs."
        # Decide if processing should stop on first error, or continue with other files
        # continue or exit 1
    fi
done

# Process NER files (Keep this analysis part as it might still be useful)
print_status "Processing NER files from: $NER_DIR"
find "$NER_DIR" -type f -name "*.json" | while IFS= read -r file; do
    # Skip __init__ files
    if [[ "$(basename "$file")" == "__init__"* ]]; then
        print_status "Skipping __init__ file: $file"
        continue
    fi
    
    print_status "Analyzing NER file: $file"
    filename=$(basename "$file")
    
    # Count entities if jq is available
    if command -v jq >/dev/null 2>&1; then
        if jq -e . "$file" > /dev/null 2>&1; then # Check if valid JSON first
            if jq -e '.entities' "$file" > /dev/null 2>&1; then
                entity_count=$(jq '.entities | length // 0' "$file")
                print_status "Found $entity_count entities in $filename"
            elif jq -e '.nodes' "$file" > /dev/null 2>&1; then
                node_count=$(jq '.nodes | length // 0' "$file")
                edge_count=$(jq '.edges | length // 0' "$file")
                print_status "Found $node_count nodes and $edge_count edges in $filename"
            else
                print_status "No recognized entity/node structure in valid JSON file $filename"
            fi
        else
            print_warning "NER file $filename is not valid JSON."
        fi
    else
         print_status "Analyzed $filename (jq not found for detailed analysis)"
    fi
done

# Process chunker files (Keep this analysis part)
print_status "Processing chunker files from: $CHUNKER_DIR"
find "$CHUNKER_DIR" -type f -name "*.json" | while IFS= read -r file; do
    # Skip __init__ files
    if [[ "$(basename "$file")" == "__init__"* ]]; then
        print_status "Skipping __init__ file: $file"
        continue
    fi
    
    print_status "Analyzing chunker file: $file"
    filename=$(basename "$file")
    
    # Count chunks if jq is available
    if command -v jq >/dev/null 2>&1; then
         if jq -e . "$file" > /dev/null 2>&1; then # Check if valid JSON first
            if jq -e '.chunks' "$file" > /dev/null 2>&1; then
                chunk_count=$(jq '.chunks | length // 0' "$file")
                print_status "Found $chunk_count chunks in $filename"
            else
                print_status "No recognized chunk structure in valid JSON file $filename"
            fi
        else
            print_warning "Chunker file $filename is not valid JSON."
        fi
    else
        print_status "Analyzed $filename (jq not found for detailed analysis)"
    fi
done

# REMOVED: Combine all entity data into a single file section
# The python script called in the loop above should now handle Neo4j loading directly.

# REMOVED: Neo4j import section
# The import logic is assumed to be handled by the python script called earlier.

print_status "Cohere pipeline processing completed."
print_status "Entity/Relationship loading to Neo4j was attempted directly during file processing."
print_status "Temporary files (if any were created by python script) might be in: $WORK_DIR"
print_status "To clean up temporary files directory, run: rm -rf $WORK_DIR"
# Optionally clean up the parent @temp if empty, but safer to leave it
# find ./@temp -maxdepth 0 -type d -empty -delete

exit 0 