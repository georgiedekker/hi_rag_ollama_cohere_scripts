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

# Load environment variables from .env file if it exists
load_env_vars() {
    if [ -f ".env" ]; then
        print_status "Loading environment variables from .env file"
        set -o allexport
        source .env
        set +o allexport
        
        # Export variables for HiRAG to use
        export OPENAI_API_KEY=${OPENAI_API_KEY:-"ollama"}
        export OPENAI_API_BASE=${OPENAI_API_BASE:-${OLLAMA_BASE_URL:-"http://localhost:11434/v1"}}
        export OPENAI_BASE_URL=${OPENAI_BASE_URL:-${OLLAMA_BASE_URL:-"http://localhost:11434/v1"}}
        export PROVIDER=${PROVIDER:-"glm"}
        
        print_status "Using API Base: $OPENAI_API_BASE"
        print_status "Selected provider: $PROVIDER"
        
        if [ ! -z "$OPENAI_MODEL" ]; then
            print_status "Using model from .env: $OPENAI_MODEL"
        fi
        
        if [ ! -z "$GLM_MODEL" ]; then
            print_status "Using GLM model from .env: $GLM_MODEL"
        fi
        
        if [ ! -z "$EMBEDDING_MODEL" ]; then
            print_status "Using embedding model from .env: $EMBEDDING_MODEL"
        fi
    elif [ -f "../.env" ]; then
        print_status "Loading environment variables from ../.env file"
        set -o allexport
        source ../.env
        set +o allexport
        
        # Export variables for HiRAG to use
        export OPENAI_API_KEY=${OPENAI_API_KEY:-"ollama"}
        export OPENAI_API_BASE=${OPENAI_API_BASE:-${OLLAMA_BASE_URL:-"http://localhost:11434/v1"}}
        export OPENAI_BASE_URL=${OPENAI_BASE_URL:-${OLLAMA_BASE_URL:-"http://localhost:11434/v1"}}
        export PROVIDER=${PROVIDER:-"glm"}
        
        print_status "Using API Base: $OPENAI_API_BASE"
        print_status "Selected provider: $PROVIDER"
        
        if [ ! -z "$OPENAI_MODEL" ]; then
            print_status "Using model from .env: $OPENAI_MODEL"
        fi
        
        if [ ! -z "$GLM_MODEL" ]; then
            print_status "Using GLM model from .env: $GLM_MODEL"
        fi
        
        if [ ! -z "$EMBEDDING_MODEL" ]; then
            print_status "Using embedding model from .env: $EMBEDDING_MODEL"
        fi
    else
        print_status "No .env file found, using environment variables"
    fi
}

# Update config.yaml from .env variables
update_config() {
    if [ -f "update_config.py" ]; then
        print_status "Updating config.yaml from .env variables"
        python update_config.py
        if [ $? -ne 0 ]; then
            print_error "Failed to update config.yaml"
            return 1
        fi
    else
        print_warning "update_config.py not found, skipping config update"
    fi
    return 0
}

# Default values
INGEST_DIR=""
NER_DIR=""
CHUNKER_DIR=""
OUTPUT_DIR=""
CONFIG="config.yaml"
QUERY=""
MODE="hi"
CLEAN=false
USE_NEO4J=true  # Default to use Neo4j
NEO4J_URL=""
NEO4J_USER=""
NEO4J_PASSWORD=""
NAIVE_RAG=false
HIERARCHICAL=true
CHUNK_SIZE=1200
CHUNK_OVERLAP=100
MAX_CLUSTER_SIZE=10
USE_HNSWLIB=false
EMBEDDING_BATCH=32
EMBEDDING_ASYNC=8
MODEL=""
PROVIDER=""

# Display usage
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -i, --ingest-dir DIR       Directory containing ingested data (required)"
    echo "  -n, --ner-dir DIR          Directory containing NER data (required)"
    echo "  -c, --chunker-dir DIR      Directory containing chunked data (required)"
    echo "  -o, --output-dir DIR       Directory to save output (defaults to hi_rag/data)"
    echo "  --config FILE              Path to config file (default: config.yaml)"
    echo "  -q, --query QUERY          Query to process"
    echo "  -m, --mode MODE            Query mode: hi, naive, hi_nobridge, hi_local, hi_global, hi_bridge (default: hi)"
    echo "  --clean                    Clean the vector database before starting"
    echo "  --use-neo4j                Use Neo4j for graph storage (default: enabled)"
    echo "  --no-neo4j                 Don't use Neo4j for graph storage"
    echo "  --neo4j-url URL            Neo4j URL (default: from env or neo4j://localhost:7687)"
    echo "  --neo4j-user USER          Neo4j username (default: from env or neo4j)"
    echo "  --neo4j-password PASS      Neo4j password (default: from env or password)"
    echo "  --naive-rag                Enable naive RAG mode"
    echo "  --no-hierarchical          Disable hierarchical mode"
    echo "  --chunk-size SIZE          Chunk token size for text splitting (default: 1200)"
    echo "  --chunk-overlap SIZE       Chunk overlap token size (default: 100)"
    echo "  --max-cluster-size SIZE    Maximum graph cluster size (default: 10)"
    echo "  --use-hnswlib              Use HNSWLib for vector database instead of NanoVectorDB"
    echo "  --embedding-batch NUM      Embedding batch number (default: 32)"
    echo "  --embedding-async NUM      Maximum concurrent embedding function calls (default: 8)"
    echo "  --model NAME               Override the model to use (default: from env or provider-specific default)"
    echo "  --provider PROVIDER        Model provider to use: openai, ollama, deepseek, azure, glm (default: from env)"
    echo "  --skip-config-update       Skip updating config.yaml from .env"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "NOTE: This script processes only the first file from each directory to reduce resource usage."
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 1
            ;;
        -i|--ingest-dir)
            INGEST_DIR="$2"
            shift 2
            ;;
        -n|--ner-dir)
            NER_DIR="$2"
            shift 2
            ;;
        -c|--chunker-dir)
            CHUNKER_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -q|--query)
            QUERY="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --use-neo4j)
            USE_NEO4J=true
            shift
            ;;
        --no-neo4j)
            USE_NEO4J=false
            shift
            ;;
        --neo4j-url)
            NEO4J_URL="$2"
            shift 2
            ;;
        --neo4j-user)
            NEO4J_USER="$2"
            shift 2
            ;;
        --neo4j-password)
            NEO4J_PASSWORD="$2"
            shift 2
            ;;
        --naive-rag)
            NAIVE_RAG=true
            shift
            ;;
        --no-hierarchical)
            HIERARCHICAL=false
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
            shift 2
            ;;
        --max-cluster-size)
            MAX_CLUSTER_SIZE="$2"
            shift 2
            ;;
        --use-hnswlib)
            USE_HNSWLIB=true
            shift
            ;;
        --embedding-batch)
            EMBEDDING_BATCH="$2"
            shift 2
            ;;
        --embedding-async)
            EMBEDDING_ASYNC="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --skip-config-update)
            SKIP_CONFIG_UPDATE=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INGEST_DIR" ] || [ -z "$NER_DIR" ] || [ -z "$CHUNKER_DIR" ]; then
    print_error "Missing required directories."
    usage
    exit 1
fi

# Load environment variables
load_env_vars

# Update config.yaml from .env (unless skipped)
if [ "$SKIP_CONFIG_UPDATE" != "true" ]; then
    update_config
fi

# Set provider from command line, environment, or default
if [ -z "$PROVIDER" ]; then
    if [ ! -z "$PROVIDER" ]; then
        print_status "Using provider from environment: $PROVIDER"
    else
        # Auto-detect provider based on environment variables
        if [ ! -z "$COHERE_API_KEY" ]; then
            PROVIDER="cohere"
            print_status "Auto-detected Cohere provider from environment variables"
        elif [ ! -z "$GLM_MODEL" ]; then
            PROVIDER="glm"
            print_status "Auto-detected GLM provider from environment variables"
        elif [ ! -z "$DEEPSEEK_API_KEY" ] && [ ! -z "$DEEPSEEK_COMPLETION_MODEL" ]; then
            PROVIDER="deepseek"
            print_status "Auto-detected DeepSeek provider from environment variables"
        elif [ ! -z "$OLLAMA_BASE_URL" ] || [[ "$OPENAI_API_BASE" == *"ollama"* ]] || [[ "$OPENAI_API_KEY" == "ollama" ]]; then
            PROVIDER="ollama"
            print_status "Auto-detected Ollama provider from environment variables"
        else
            PROVIDER="openai"
            print_status "Using default OpenAI provider"
        fi
    fi
else
    print_status "Using provider from command line: $PROVIDER"
fi

# Set model from command line or environment
if [ -z "$MODEL" ]; then
    if [ "$PROVIDER" = "cohere" ] && [ ! -z "$COHERE_EMBEDDING_MODEL" ]; then
        MODEL="$COHERE_EMBEDDING_MODEL"
        print_status "Using Cohere model from environment: $MODEL"
    elif [ "$PROVIDER" = "glm" ] && [ ! -z "$GLM_MODEL" ]; then
        MODEL="$GLM_MODEL"
        print_status "Using GLM model from environment: $MODEL"
    elif [ "$PROVIDER" = "deepseek" ] && [ ! -z "$DEEPSEEK_COMPLETION_MODEL" ]; then
        MODEL="$DEEPSEEK_COMPLETION_MODEL"
        print_status "Using DeepSeek model from environment: $MODEL"
    elif [ ! -z "$OPENAI_MODEL" ]; then
        MODEL="$OPENAI_MODEL"
        print_status "Using model from environment: $MODEL"
    fi
else
    print_status "Using model from command line: $MODEL"
fi

# Create temp directories for single file processing
print_status "Creating temporary directories with single files..."

TEMP_DIR=$(mktemp -d)
print_status "Created temporary directory: $TEMP_DIR"

# Create subdirectories
TEMP_INGEST_DIR="$TEMP_DIR/ingest"
TEMP_NER_DIR="$TEMP_DIR/ner"
TEMP_CHUNKER_DIR="$TEMP_DIR/chunker"
mkdir -p "$TEMP_INGEST_DIR" "$TEMP_NER_DIR" "$TEMP_CHUNKER_DIR"

# Find first relevant file in each source directory
print_status "Finding first file in ingest directory..."
print_status "Listing ingest directory content for debugging:"
find "$INGEST_DIR" -type f | sort | head -n 10

# First try with strict filtering
FIRST_INGEST_FILE=$(find "$INGEST_DIR" -type f -not -name ".*" -not -name ".DS_Store" | grep -v "^\." | head -n 1)

# If nothing found, try with less strict filtering including subdirectories
if [ -z "$FIRST_INGEST_FILE" ]; then
    print_status "No files found with strict filtering, trying with less strict criteria..."
    FIRST_INGEST_FILE=$(find "$INGEST_DIR" -type f -not -name ".DS_Store" | head -n 1)
    
    # If still nothing found, try including all files
    if [ -z "$FIRST_INGEST_FILE" ]; then
        print_status "Trying with minimal filtering..."
        FIRST_INGEST_FILE=$(find "$INGEST_DIR" -type f | head -n 1)
        
        # If still nothing, look one level deeper for subdirectories
        if [ -z "$FIRST_INGEST_FILE" ]; then
            print_status "Looking in subdirectories..."
            FIRST_INGEST_DIR=$(find "$INGEST_DIR" -type d | head -n 2 | tail -n 1)
            if [ ! -z "$FIRST_INGEST_DIR" ] && [ "$FIRST_INGEST_DIR" != "$INGEST_DIR" ]; then
                FIRST_INGEST_FILE=$(find "$FIRST_INGEST_DIR" -type f | head -n 1)
                print_status "Found subdirectory: $FIRST_INGEST_DIR"
            fi
        fi
    fi
fi

if [ -z "$FIRST_INGEST_FILE" ]; then
    print_error "No valid files found in ingest directory: $INGEST_DIR"
    exit 1
fi
print_status "Selected ingest file: $FIRST_INGEST_FILE"

print_status "Finding first file in NER directory..."
print_status "Listing NER directory content for debugging:"
find "$NER_DIR" -type f | sort | head -n 10

# First try with strict filtering
FIRST_NER_FILE=$(find "$NER_DIR" -type f -not -name ".*" -not -name ".DS_Store" | grep -v "^\." | head -n 1)

# If nothing found, try with less strict filtering including subdirectories
if [ -z "$FIRST_NER_FILE" ]; then
    print_status "No files found with strict filtering, trying with less strict criteria..."
    FIRST_NER_FILE=$(find "$NER_DIR" -type f -not -name ".DS_Store" | head -n 1)
    
    # If still nothing found, try including all files
    if [ -z "$FIRST_NER_FILE" ]; then
        print_status "Trying with minimal filtering..."
        FIRST_NER_FILE=$(find "$NER_DIR" -type f | head -n 1)
        
        # If still nothing, look one level deeper for subdirectories
        if [ -z "$FIRST_NER_FILE" ]; then
            print_status "Looking in subdirectories..."
            FIRST_NER_DIR=$(find "$NER_DIR" -type d | head -n 2 | tail -n 1)
            if [ ! -z "$FIRST_NER_DIR" ] && [ "$FIRST_NER_DIR" != "$NER_DIR" ]; then
                FIRST_NER_FILE=$(find "$FIRST_NER_DIR" -type f | head -n 1)
                print_status "Found subdirectory: $FIRST_NER_DIR"
            fi
        fi
    fi
fi

if [ -z "$FIRST_NER_FILE" ]; then
    print_error "No valid files found in NER directory: $NER_DIR"
    exit 1
fi
print_status "Selected NER file: $FIRST_NER_FILE"

print_status "Finding first file in chunker directory..."
print_status "Listing chunker directory content for debugging:"
find "$CHUNKER_DIR" -type f | sort | head -n 10

# First try with strict filtering
FIRST_CHUNKER_FILE=$(find "$CHUNKER_DIR" -type f -not -name ".*" -not -name ".DS_Store" | grep -v "^\." | head -n 1)

# If nothing found, try with less strict filtering including subdirectories
if [ -z "$FIRST_CHUNKER_FILE" ]; then
    print_status "No files found with strict filtering, trying with less strict criteria..."
    FIRST_CHUNKER_FILE=$(find "$CHUNKER_DIR" -type f -not -name ".DS_Store" | head -n 1)
    
    # If still nothing found, try including all files
    if [ -z "$FIRST_CHUNKER_FILE" ]; then
        print_status "Trying with minimal filtering..."
        FIRST_CHUNKER_FILE=$(find "$CHUNKER_DIR" -type f | head -n 1)
        
        # If still nothing, look one level deeper for subdirectories
        if [ -z "$FIRST_CHUNKER_FILE" ]; then
            print_status "Looking in subdirectories..."
            FIRST_CHUNKER_DIR=$(find "$CHUNKER_DIR" -type d | head -n 2 | tail -n 1)
            if [ ! -z "$FIRST_CHUNKER_DIR" ] && [ "$FIRST_CHUNKER_DIR" != "$CHUNKER_DIR" ]; then
                FIRST_CHUNKER_FILE=$(find "$FIRST_CHUNKER_DIR" -type f | head -n 1)
                print_status "Found subdirectory: $FIRST_CHUNKER_DIR"
            fi
        fi
    fi
fi

if [ -z "$FIRST_CHUNKER_FILE" ]; then
    print_error "No valid files found in chunker directory: $CHUNKER_DIR"
    exit 1
fi
print_status "Selected chunker file: $FIRST_CHUNKER_FILE"

# Copy first files to temporary directories
cp "$FIRST_INGEST_FILE" "$TEMP_INGEST_DIR/"
cp "$FIRST_NER_FILE" "$TEMP_NER_DIR/"
cp "$FIRST_CHUNKER_FILE" "$TEMP_CHUNKER_DIR/"

print_status "Files copied to temporary directories. Starting pipeline with single file..."

# Print configuration
echo "[PIPELINE] Starting pipeline integration with single file..."
echo "[PIPELINE] Temp ingest directory: $TEMP_INGEST_DIR"
echo "[PIPELINE] Temp NER directory: $TEMP_NER_DIR"
echo "[PIPELINE] Temp chunker directory: $TEMP_CHUNKER_DIR"
echo "[PIPELINE] Provider: $PROVIDER"
echo "[PIPELINE] Model: $MODEL"

# Build the command
CMD="python pipeline_integration.py --ingest-dir \"$TEMP_INGEST_DIR\" --ner-dir \"$TEMP_NER_DIR\" --chunker-dir \"$TEMP_CHUNKER_DIR\" --config \"$CONFIG\""

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
else
    # Default output directory in temp location
    TEMP_OUTPUT_DIR="$TEMP_DIR/output"
    mkdir -p "$TEMP_OUTPUT_DIR"
    CMD="$CMD --output-dir \"$TEMP_OUTPUT_DIR\""
    print_status "Using temporary output directory: $TEMP_OUTPUT_DIR"
fi

if [ ! -z "$QUERY" ]; then
    CMD="$CMD --query \"$QUERY\""
fi

if [ "$CLEAN" = true ]; then
    CMD="$CMD --clean"
fi

if [ "$USE_NEO4J" = true ]; then
    echo "[PIPELINE] Using Neo4j for graph storage"
    CMD="$CMD --use-neo4j"
    
    if [ ! -z "$NEO4J_URL" ]; then
        echo "[PIPELINE] Neo4j URL: $NEO4J_URL"
        CMD="$CMD --neo4j-url \"$NEO4J_URL\""
    fi
    
    if [ ! -z "$NEO4J_USER" ]; then
        echo "[PIPELINE] Neo4j User: $NEO4J_USER"
        CMD="$CMD --neo4j-user \"$NEO4J_USER\""
    fi
    
    if [ ! -z "$NEO4J_PASSWORD" ]; then
        echo "[PIPELINE] Neo4j Password: [REDACTED]"
        CMD="$CMD --neo4j-password \"$NEO4J_PASSWORD\""
    fi
fi

if [ "$NAIVE_RAG" = true ]; then
    CMD="$CMD --naive-rag"
fi

if [ "$HIERARCHICAL" = false ]; then
    CMD="$CMD --no-hierarchical"
fi

if [ "$CHUNK_SIZE" != "1200" ]; then
    CMD="$CMD --chunk-size $CHUNK_SIZE"
fi

if [ "$CHUNK_OVERLAP" != "100" ]; then
    CMD="$CMD --chunk-overlap $CHUNK_OVERLAP"
fi

if [ "$MAX_CLUSTER_SIZE" != "10" ]; then
    CMD="$CMD --max-cluster-size $MAX_CLUSTER_SIZE"
fi

if [ "$USE_HNSWLIB" = true ]; then
    CMD="$CMD --use-hnswlib"
fi

if [ "$EMBEDDING_BATCH" != "32" ]; then
    CMD="$CMD --embedding-batch $EMBEDDING_BATCH"
fi

if [ "$EMBEDDING_ASYNC" != "8" ]; then
    CMD="$CMD --embedding-async $EMBEDDING_ASYNC"
fi

if [ ! -z "$MODEL" ]; then
    echo "[PIPELINE] Using model: $MODEL"
    CMD="$CMD --model \"$MODEL\""
fi

if [ ! -z "$PROVIDER" ]; then
    echo "[PIPELINE] Using provider: $PROVIDER"
    CMD="$CMD --provider \"$PROVIDER\""
fi

# Run the command
echo "[PIPELINE] Running pipeline integration command with single file..."
echo $CMD
eval $CMD

# Check the exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "[ERROR] Pipeline integration failed with exit code $EXIT_CODE."
    echo "[CLEANUP] Removing temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
    exit $EXIT_CODE
else
    echo "[SUCCESS] Pipeline integration completed successfully."
    echo "[CLEANUP] Removing temporary directory: $TEMP_DIR"
    rm -rf "$TEMP_DIR"
fi 
exit 0 