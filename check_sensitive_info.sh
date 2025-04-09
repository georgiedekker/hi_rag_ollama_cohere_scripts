#!/bin/bash

# Set colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Checking for sensitive information in the codebase...${NC}"

# Create a temporary file for findings
FINDINGS_FILE=$(mktemp)

# Define patterns to search for
PATTERNS=(
    "api[_-]key"
    "secret"
    "password"
    "token"
    "auth"
    "credential"
    "bearer"
    "sk-[a-zA-Z0-9]{24,96}" # OpenAI API key pattern
    "-----BEGIN PRIVATE KEY-----" # Private key pattern
    "[a-zA-Z0-9+/]{50,}=" # Base64 encoded secrets
)

# Files to exclude from checking
EXCLUDES=(
    ".env"
    ".env.*"
    "*.env"
    ".git/*"
    "__pycache__/*"
    "*.pyc"
    "check_sensitive_info.sh"
    ".gitignore"
    "*/.git/*"
    "data/*"
    "*/data/*"
    "output/*"
    "*/output/*"
    "*.vector"
    "*.hnsw"
    "*.idx"
    "*.json"
    "REPOSITORY.md"
    "README.md"
)

# Construct exclude arguments
EXCLUDE_ARGS=""
for pattern in "${EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pattern"
done

# Only search in specific file types that will be published
FILE_TYPES=(
    "*.py"
    "*.sh"
    "*.yaml"
    "*.yml"
    "Dockerfile"
    "docker-compose*"
    ".env.example"
)

# Construct file type arguments
FILE_TYPE_ARGS=""
for file_type in "${FILE_TYPES[@]}"; do
    FILE_TYPE_ARGS="$FILE_TYPE_ARGS --include=$file_type"
done

# Search for each pattern
for pattern in "${PATTERNS[@]}"; do
    echo -e "Searching for pattern: ${pattern}"
    grep -r -i -n $EXCLUDE_ARGS $FILE_TYPE_ARGS --include="*.py" --include="*.sh" --include="*.yaml" --include="*.yml" "$pattern" . >> "$FINDINGS_FILE"
done

# Check .env files
ENV_FILES=$(find . -name ".env*" | grep -v ".env.example")
if [ -n "$ENV_FILES" ]; then
    echo -e "${RED}WARNING: Found .env files that might contain sensitive information:${NC}"
    echo "$ENV_FILES"
    echo -e ".env files should be excluded from Git. Make sure they're in .gitignore.\n" >> "$FINDINGS_FILE"
fi

# Check if any findings were found
if [ -s "$FINDINGS_FILE" ]; then
    echo -e "${RED}WARNING: Potentially sensitive information found:${NC}"
    cat "$FINDINGS_FILE"
    echo ""
    echo -e "${RED}Please review these findings before committing to Git.${NC}"
    echo -e "Suggestions:"
    echo -e "1. Replace hard-coded credentials with environment variables."
    echo -e "2. Use placeholders in example files."
    echo -e "3. Verify .gitignore is correctly set up."
    exit 1
else
    echo -e "${GREEN}No sensitive information detected!${NC}"
    echo -e "It's still recommended to manually review your code before pushing to Git."
fi

# Clean up
rm "$FINDINGS_FILE"

echo -e "${YELLOW}Checking for configuration files...${NC}"
CONFIG_FILES=$(find . -name "config.yaml" -o -name "*.config.*" | grep -v "example" | grep -v "template")

if [ -n "$CONFIG_FILES" ]; then
    echo -e "${YELLOW}Found configuration files that should be checked manually:${NC}"
    echo "$CONFIG_FILES"
    echo -e "Make sure these files don't contain hard-coded API keys or credentials."
fi

echo -e "${GREEN}Check completed. Ready for Git publishing.${NC}"

exit 0 