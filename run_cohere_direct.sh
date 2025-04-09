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
        print_status "No .env file found in current or parent directory"
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

# Set the chat model if not already set
if [ -z "$COHERE_CHAT_MODEL" ]; then
    export COHERE_CHAT_MODEL="command"
    print_status "Using default Cohere chat model: $COHERE_CHAT_MODEL"
else
    print_status "Using Cohere chat model from environment: $COHERE_CHAT_MODEL"
fi

# Create temporary directory for output
TEMP_DIR=$(mktemp -d)
print_status "Created temporary directory: $TEMP_DIR"

# Set up the file paths
INPUT_FILE="$1"
if [ -z "$INPUT_FILE" ]; then
    print_error "No input file provided. Usage: $0 input_file"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    print_error "Input file does not exist: $INPUT_FILE"
    exit 1
fi

OUTPUT_FILE="$TEMP_DIR/extracted_entities.json"

print_status "Input file: $INPUT_FILE"
print_status "Output file: $OUTPUT_FILE"

# Run the entity extraction script
print_status "Running Cohere entity extraction..."
python mini_entity_extract.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$COHERE_CHAT_MODEL"

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    print_error "Entity extraction failed with exit code $EXIT_CODE"
    print_status "Removing temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
    exit $EXIT_CODE
fi

# Show the results
print_success "Entity extraction completed successfully"
print_status "Extracted entities saved to: $OUTPUT_FILE"
print_status "Entity count:"
jq '.entities | length' "$OUTPUT_FILE"
print_status "Relationship count:"
jq '.relationships | length' "$OUTPUT_FILE"

# Keep the output directory for user to examine
print_status "Results are available in: $OUTPUT_FILE"

exit 0 