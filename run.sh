#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${YELLOW}[HI_RAG]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help message
show_help() {
    echo "HI_RAG - Hierarchical Retrieval Augmented Generation"
    echo ""
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help                Show this help message"
    echo "  -s, --setup               Run the setup verification script"
    echo "  -q, --query \"QUERY\"       Run a query (required if not using --setup)"
    echo "  -d, --document FILE       Specify document to index (default: sample_document.txt)"
    echo "  -m, --mode MODE           Query mode: hi, naive, hi_nobridge, hi_local, hi_global, hi_bridge (default: hi)"
    echo "  -f, --force-reindex       Force reindexing even if vector store exists"
    echo "  -c, --clean               Clean the vector database before starting"
    echo "  -t, --test                Run tests"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --setup                     # Verify and setup the environment"
    echo "  ./run.sh -q \"What is HiRAG?\"         # Run a query with default settings"
    echo "  ./run.sh -q \"What is HiRAG?\" -m naive -f  # Run with naive mode and force reindexing"
    echo "  ./run.sh -q \"What is HiRAG?\" -c     # Clean vector database before running"
    echo "  ./run.sh -t                          # Run tests"
}

# Default values
QUERY=""
DOCUMENT="sample_document.txt"
MODE="hi"
FORCE_REINDEX=""
CLEAN_DB=""
RUN_SETUP=false
RUN_TESTS=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--setup)
            RUN_SETUP=true
            shift
            ;;
        -q|--query)
            QUERY="$2"
            shift 2
            ;;
        -d|--document)
            DOCUMENT="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -f|--force-reindex)
            FORCE_REINDEX="--force-reindex"
            shift
            ;;
        -c|--clean)
            CLEAN_DB="--clean"
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we're in the right directory
if [[ ! -f "run_hirag.py" && ! -f "verify_setup.py" ]]; then
    # Try to find the hi_rag directory
    if [[ -d "../hi_rag" ]]; then
        cd ../hi_rag
    elif [[ -d "hi_rag" ]]; then
        cd hi_rag
    else
        print_error "Could not find the hi_rag directory. Please run this script from the hi_rag directory."
        exit 1
    fi
fi

# Run setup verification if requested
if [[ "$RUN_SETUP" == true ]]; then
    print_status "Running setup verification..."
    python verify_setup.py
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Setup verification completed successfully."
    else
        print_error "Setup verification failed."
        exit $exit_code
    fi
fi

# Run tests if requested
if [[ "$RUN_TESTS" == true ]]; then
    print_status "Running tests..."
    python test_hirag.py
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Tests completed successfully."
    else
        print_error "Tests failed."
        exit $exit_code
    fi
fi

# Exit if no query is provided and we're not running setup or tests
if [[ -z "$QUERY" && "$RUN_SETUP" == false && "$RUN_TESTS" == false ]]; then
    print_error "No action specified. Please provide a query or use --setup or --test."
    show_help
    exit 1
fi

# Run query if provided
if [[ ! -z "$QUERY" ]]; then
    print_status "Running query: \"$QUERY\" (mode: $MODE)"
    print_status "Document: $DOCUMENT"
    
    # Build the command
    CMD="python run_hirag.py --query \"$QUERY\" --mode $MODE --document \"$DOCUMENT\" $FORCE_REINDEX $CLEAN_DB"
    
    # Run the command
    eval $CMD
    exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        print_success "Query completed successfully."
    else
        print_error "Query failed with exit code $exit_code."
        exit $exit_code
    fi
fi

exit 0 