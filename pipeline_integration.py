#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
import logging
import shutil
import glob
import yaml
from pathlib import Path
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv

# IMPORTANT: Remove this import to avoid loading the demo module
# from hi_rag_demo import embedding_function

# Import hi_rag components directly
from hirag import HiRAG, QueryParam
from hirag._storage import NetworkXStorage, Neo4jStorage, JsonKVStorage, NanoVectorDBStorage, HNSWVectorStorage

# Define the embedding function directly here instead of importing from demo
def embedding_function(texts: List[str]) -> np.ndarray:
    """Simple wrapper for the embedding function to handle batch processing"""
    # Create a dummy embedding function that returns random vectors
    # This is just a fallback - actual embeddings will be handled by HiRAG directly
    embedding_dim = 3584  # Default dimension for gte models
    return np.random.rand(len(texts), embedding_dim).astype(np.float32)

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize environment variables with values from .env
def initialize_env_variables():
    """Initialize and verify environment variables from .env file"""
    # Required for Ollama
    if 'OPENAI_API_BASE' not in os.environ and 'OPENAI_BASE_URL' not in os.environ:
        logger.warning("Missing OPENAI_API_BASE or OPENAI_BASE_URL in environment variables. Setting to default Ollama URL.")
        os.environ['OPENAI_API_BASE'] = "http://localhost:11434/v1"
    
    if 'OPENAI_API_KEY' not in os.environ:
        logger.warning("Missing OPENAI_API_KEY in environment variables. Setting to 'ollama' for Ollama compatibility.")
        os.environ['OPENAI_API_KEY'] = "ollama"
    
    # Provider preference
    if 'PROVIDER' not in os.environ:
        # Auto-detect provider
        if 'GLM_MODEL' in os.environ:
            os.environ['PROVIDER'] = 'glm'
            logger.info(f"Setting PROVIDER to 'glm' based on GLM_MODEL in .env")
        elif 'DEEPSEEK_API_KEY' in os.environ and os.environ.get('DEEPSEEK_API_KEY'):
            os.environ['PROVIDER'] = 'deepseek'
            logger.info(f"Setting PROVIDER to 'deepseek' based on DEEPSEEK_API_KEY in .env")
        elif os.environ.get('OPENAI_API_BASE', '').find('ollama') >= 0 or os.environ.get('OPENAI_API_KEY') == 'ollama':
            os.environ['PROVIDER'] = 'ollama'
            logger.info(f"Setting PROVIDER to 'ollama' based on API configuration in .env")
        else:
            os.environ['PROVIDER'] = 'openai'
            logger.info(f"Setting PROVIDER to 'openai' as default")
    else:
        logger.info(f"Using PROVIDER from .env: {os.environ['PROVIDER']}")
    
    # Ensure OPENAI_MODEL_NAME is set from appropriate source
    if 'OPENAI_MODEL_NAME' not in os.environ:
        provider = os.environ.get('PROVIDER', '').lower()
        if provider == 'glm' and 'GLM_MODEL' in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = os.environ['GLM_MODEL']
            logger.info(f"Setting OPENAI_MODEL_NAME to GLM_MODEL: {os.environ['OPENAI_MODEL_NAME']}")
        elif provider == 'deepseek' and 'DEEPSEEK_COMPLETION_MODEL' in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = os.environ['DEEPSEEK_COMPLETION_MODEL']
            logger.info(f"Setting OPENAI_MODEL_NAME to DEEPSEEK_COMPLETION_MODEL: {os.environ['OPENAI_MODEL_NAME']}")
        elif 'OPENAI_MODEL' in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = os.environ['OPENAI_MODEL']
            logger.info(f"Setting OPENAI_MODEL_NAME to OPENAI_MODEL: {os.environ['OPENAI_MODEL_NAME']}")
        else:
            # Default value based on provider
            if provider == 'glm':
                os.environ['OPENAI_MODEL_NAME'] = 'glm4'
            elif provider == 'deepseek':
                os.environ['OPENAI_MODEL_NAME'] = 'deepseek-chat'
            elif provider == 'ollama':
                os.environ['OPENAI_MODEL_NAME'] = 'llama3'
            else:
                os.environ['OPENAI_MODEL_NAME'] = 'gpt-4o-mini'
            logger.warning(f"No model specified in .env, using default for {provider}: {os.environ['OPENAI_MODEL_NAME']}")
    else:
        logger.info(f"Using OPENAI_MODEL_NAME from environment: {os.environ['OPENAI_MODEL_NAME']}")
    
    # Setting EMBEDDING_MODEL if not set
    if 'EMBEDDING_MODEL' not in os.environ and 'OPENAI_MODEL' in os.environ:
        os.environ['EMBEDDING_MODEL'] = os.environ['OPENAI_MODEL']
        logger.info(f"Setting EMBEDDING_MODEL to OPENAI_MODEL: {os.environ['EMBEDDING_MODEL']}")

# Initialize environment variables
initialize_env_variables()

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable substitution"""
    try:
        with open(config_path, 'r') as file:
            # Read the file content
            config_str = file.read()
            
            # Substitute environment variables
            for key, value in os.environ.items():
                if value is not None:
                    config_str = config_str.replace(f'${{{key}}}', value)
            
            # Load YAML from the substituted string
            config = yaml.safe_load(config_str)
            
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def setup_provider_config(config: Dict[str, Any], provider: str = None, model: Optional[str] = None) -> bool:
    """Set up environment variables for the selected provider"""
    # Use provider from arguments, environment, or auto-detect
    if provider is None:
        provider = os.environ.get('PROVIDER', '').lower()
    else:
        provider = provider.lower()
    
    logger.info(f"Setting up environment for provider: {provider}")
    
    # Set the PROVIDER environment variable
    os.environ['PROVIDER'] = provider
    
    if provider == "openai":
        # OpenAI configuration
        os.environ['OPENAI_API_KEY'] = config.get('openai', {}).get('api_key', os.environ.get('OPENAI_API_KEY', ''))
        os.environ['OPENAI_BASE_URL'] = config.get('openai', {}).get('base_url', os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        
        # Set model if provided via argument, otherwise use from environment
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            logger.info(f"Overriding model with argument: {model}")
        
        logger.info(f"Using model: {os.environ.get('OPENAI_MODEL_NAME')}")
        
    elif provider == "ollama":
        # Ollama configuration - use values from .env if available
        os.environ['OLLAMA_BASE_URL'] = os.environ.get('OLLAMA_BASE_URL', config.get('ollama', {}).get('base_url', 'http://localhost:11434'))
        os.environ['OPENAI_API_BASE'] = f"{os.environ['OLLAMA_BASE_URL']}/v1"
        os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', "ollama")
        
        # Set model if provided via argument, otherwise use from environment
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            logger.info(f"Overriding model with argument: {model}")
        
        logger.info(f"Using Ollama model: {os.environ.get('OPENAI_MODEL_NAME')}")
        
    elif provider == "deepseek":
        # DeepSeek configuration - prioritize values from .env
        os.environ['DEEPSEEK_API_KEY'] = os.environ.get('DEEPSEEK_API_KEY', config.get('deepseek', {}).get('api_key', ''))
        os.environ['DEEPSEEK_BASE_URL'] = os.environ.get('DEEPSEEK_BASE_URL', config.get('deepseek', {}).get('base_url', 'https://api.deepseek.com'))
        
        # Set model if provided via argument, otherwise use DEEPSEEK_COMPLETION_MODEL
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            os.environ['DEEPSEEK_MODEL'] = model
            logger.info(f"Overriding model with argument: {model}")
        elif 'DEEPSEEK_COMPLETION_MODEL' in os.environ and 'OPENAI_MODEL_NAME' not in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = os.environ['DEEPSEEK_COMPLETION_MODEL']
            os.environ['DEEPSEEK_MODEL'] = os.environ['DEEPSEEK_COMPLETION_MODEL']
            
        logger.info(f"Using DeepSeek model: {os.environ.get('DEEPSEEK_MODEL', os.environ.get('OPENAI_MODEL_NAME'))}")
        
    elif provider == "glm":
        # GLM configuration - prioritize GLM_MODEL from .env
        os.environ['OPENAI_API_BASE'] = os.environ.get('OPENAI_API_BASE', config.get('glm', {}).get('base_url', 'http://localhost:11434/v1'))
        os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI_API_KEY', config.get('glm', {}).get('api_key', 'ollama'))
        
        # Set model if provided via argument, otherwise use GLM_MODEL from .env
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            logger.info(f"Overriding model with argument: {model}")
        elif 'GLM_MODEL' in os.environ and 'OPENAI_MODEL_NAME' not in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = os.environ['GLM_MODEL']
            
        logger.info(f"Using GLM model: {os.environ.get('OPENAI_MODEL_NAME')}")
        
    elif provider == "azure":
        # Azure OpenAI configuration - environment variables take precedence
        os.environ['AZURE_OPENAI_API_KEY'] = os.environ.get('AZURE_OPENAI_API_KEY', config.get('azure', {}).get('api_key', ''))
        os.environ['AZURE_OPENAI_ENDPOINT'] = os.environ.get('AZURE_OPENAI_ENDPOINT', config.get('azure', {}).get('endpoint', ''))
        os.environ['AZURE_OPENAI_API_VERSION'] = os.environ.get('AZURE_OPENAI_API_VERSION', config.get('azure', {}).get('api_version', '2023-05-15'))
        
        # Set model if provided via argument, otherwise use from environment
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            logger.info(f"Overriding model with argument: {model}")
        
        logger.info(f"Using Azure model: {os.environ.get('OPENAI_MODEL_NAME')}")
        
    elif provider == "cohere":
        # Cohere configuration - prioritize cohere_embedding_dim from .env
        os.environ['COHERE_API_KEY'] = os.environ.get('COHERE_API_KEY', config.get('cohere', {}).get('api_key', ''))
        os.environ['COHERE_BASE_URL'] = os.environ.get('COHERE_BASE_URL', config.get('cohere', {}).get('base_url', 'https://api.cohere.com'))
        
        # Set model if provided via argument, otherwise use cohere_embedding_dim from .env
        if model:
            os.environ['OPENAI_MODEL_NAME'] = model
            logger.info(f"Overriding model with argument: {model}")
        elif 'cohere_embedding_dim' in config.get('model_params', {}) and 'OPENAI_MODEL_NAME' not in os.environ:
            os.environ['OPENAI_MODEL_NAME'] = str(config['model_params']['cohere_embedding_dim'])
            logger.warning(f"No model specified in .env, using default for cohere: {os.environ['OPENAI_MODEL_NAME']}")
        
        logger.info(f"Using Cohere model: {os.environ.get('OPENAI_MODEL_NAME')}")
        
    else:
        logger.error(f"Unknown provider: {provider}")
        return False
    
    return True

def clean_vector_database(data_dir: str) -> bool:
    """Clean the vector database files to fix dimension mismatch"""
    logger.info(f"Cleaning vector database in {data_dir}")
    
    # Files to delete
    vdb_files = [
        'vdb_entities.json',
        'vdb_entities.index',
        'vdb_chunks.json',
        'vdb_chunks.index',
        'graph_chunk_entity_relation.graphml',
        'graph_chunk_entity_relation.gra',
        'text_chunks.json',
        'full_docs.json',
        'community_reports.json',
        'llm_response_cache.json'
    ]
    
    for file in vdb_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    
    # Also clean any other files that might be related
    for pattern in ['*.index', '*.gra', '*.graphml']:
        for file_path in glob.glob(os.path.join(data_dir, pattern)):
            try:
                os.remove(file_path)
                logger.info(f"Deleted {file_path}")
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    
    logger.info("Vector database cleaned successfully")
    return True

def load_chunked_data(chunker_output_dir: str) -> Union[Dict[str, Any], str, None]:
    """Load chunked data from the rag_chunker output directory"""
    logger.info(f"Loading chunked data from {chunker_output_dir}")
    
    try:
        # Find and load the main output file(s)
        graph_output_path = os.path.join(chunker_output_dir, "graph_output.json")
        if os.path.exists(graph_output_path):
            with open(graph_output_path, 'r', encoding='utf-8') as f:
                chunked_data = json.load(f)
            logger.info(f"Loaded chunked data from {graph_output_path}")
            return chunked_data
        
        # If the main output file doesn't exist, look for other JSON files
        json_files = glob.glob(os.path.join(chunker_output_dir, "**/*.json"), recursive=True)
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files in chunker output directory")
            combined_data = ""
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                    
                    # Extract text content from the JSON data
                    if isinstance(file_data, dict):
                        if "chunks" in file_data:
                            # Extract text from chunks
                            for chunk in file_data["chunks"]:
                                if isinstance(chunk, dict) and "text" in chunk:
                                    combined_data += chunk["text"] + "\n\n"
                        elif "text" in file_data:
                            combined_data += file_data["text"] + "\n\n"
                    elif isinstance(file_data, list):
                        # Try to extract text from each item in the list
                        for item in file_data:
                            if isinstance(item, dict) and "text" in item:
                                combined_data += item["text"] + "\n\n"
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {e}")
            
            if combined_data:
                logger.info(f"Combined {len(combined_data)} characters of text data")
                return combined_data
            else:
                logger.warning(f"No text data found in JSON files")
                return None
        
        logger.warning(f"No chunked data found in {chunker_output_dir}")
        return None
    except Exception as e:
        logger.error(f"Error loading chunked data: {e}")
        return None

def load_ner_data(ner_output_dir: str) -> Optional[str]:
    """Load named entity recognition data from the graph_ner output directory"""
    logger.info(f"Loading NER data from {ner_output_dir}")
    
    try:
        # Find and load entity JSON files
        entity_files = glob.glob(os.path.join(ner_output_dir, "*entities*.json"))
        entity_files.extend(glob.glob(os.path.join(ner_output_dir, "**/*entities*.json"), recursive=True))
        
        if entity_files:
            logger.info(f"Found {len(entity_files)} entity files in NER output directory")
            combined_data = ""
            
            for entity_file in entity_files:
                try:
                    with open(entity_file, 'r', encoding='utf-8') as f:
                        entity_data = json.load(f)
                    
                    # Extract entity information and format it as text
                    if isinstance(entity_data, dict):
                        for key, value in entity_data.items():
                            combined_data += f"{key}: {value}\n"
                    elif isinstance(entity_data, list):
                        for entity in entity_data:
                            if isinstance(entity, dict):
                                combined_data += f"Entity: {json.dumps(entity)}\n"
                except Exception as e:
                    logger.error(f"Error loading entity file {entity_file}: {e}")
            
            if combined_data:
                logger.info(f"Combined {len(combined_data)} characters of entity data")
                return combined_data
            else:
                logger.warning(f"No entity data found in entity files")
                return None
        
        # Check for graph JSON files
        graph_files = glob.glob(os.path.join(ner_output_dir, "*graph*.json"))
        graph_files.extend(glob.glob(os.path.join(ner_output_dir, "**/*graph*.json"), recursive=True))
        
        if graph_files:
            logger.info(f"Found {len(graph_files)} graph files in NER output directory")
            combined_data = ""
            
            for graph_file in graph_files:
                try:
                    with open(graph_file, 'r', encoding='utf-8') as f:
                        graph_data = json.load(f)
                    
                    # Extract graph information and format it as text
                    if isinstance(graph_data, dict):
                        if "nodes" in graph_data:
                            for node in graph_data["nodes"]:
                                combined_data += f"Node: {json.dumps(node)}\n"
                        if "edges" in graph_data:
                            for edge in graph_data["edges"]:
                                combined_data += f"Edge: {json.dumps(edge)}\n"
                    elif isinstance(graph_data, list):
                        for item in graph_data:
                            if isinstance(item, dict):
                                combined_data += f"Item: {json.dumps(item)}\n"
                except Exception as e:
                    logger.error(f"Error loading graph file {graph_file}: {e}")
            
            if combined_data:
                logger.info(f"Combined {len(combined_data)} characters of graph data")
                return combined_data
            else:
                logger.warning(f"No graph data found in graph files")
                return None
        
        logger.warning(f"No NER data found in {ner_output_dir}")
        return None
    except Exception as e:
        logger.error(f"Error loading NER data: {e}")
        return None

def load_ingested_data(ingest_output_dir: str) -> Optional[str]:
    """Load ingested data from the ingest output directory"""
    logger.info(f"Loading ingested data from {ingest_output_dir}")
    
    try:
        # Look for text files first (most likely to contain the original document text)
        text_files = glob.glob(os.path.join(ingest_output_dir, "*.txt"))
        text_files.extend(glob.glob(os.path.join(ingest_output_dir, "**/*.txt"), recursive=True))
        
        if text_files:
            logger.info(f"Found {len(text_files)} text files in ingest output directory")
            combined_data = ""
            
            for text_file in text_files:
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_data = f.read()
                    combined_data += text_data + "\n\n"
                except Exception as e:
                    logger.error(f"Error loading text file {text_file}: {e}")
            
            if combined_data:
                logger.info(f"Combined {len(combined_data)} characters of text data")
                return combined_data
            else:
                logger.warning(f"No text data found in text files")
        
        # If no text files found, look for JSON files
        json_files = glob.glob(os.path.join(ingest_output_dir, "*.json"))
        json_files.extend(glob.glob(os.path.join(ingest_output_dir, "**/*.json"), recursive=True))
        
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files in ingest output directory")
            combined_data = ""
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Extract text content from the JSON data
                    if isinstance(json_data, dict):
                        if "text" in json_data:
                            combined_data += json_data["text"] + "\n\n"
                        elif "content" in json_data:
                            combined_data += json_data["content"] + "\n\n"
                    elif isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict):
                                if "text" in item:
                                    combined_data += item["text"] + "\n\n"
                                elif "content" in item:
                                    combined_data += item["content"] + "\n\n"
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {e}")
            
            if combined_data:
                logger.info(f"Combined {len(combined_data)} characters of JSON data")
                return combined_data
            else:
                logger.warning(f"No text data found in JSON files")
        
        logger.warning(f"No ingested data found in {ingest_output_dir}")
        return None
    except Exception as e:
        logger.error(f"Error loading ingested data: {e}")
        return None

def combine_data(chunked_data, ner_data, ingested_data) -> str:
    """Combine data from different sources"""
    logger.info("Combining data from different sources")
    
    combined_data = ""
    
    # Add metadata section
    combined_data += "# Document Metadata\n\n"
    
    # Add chunked data
    if chunked_data:
        combined_data += "## Chunked Data\n\n"
        if isinstance(chunked_data, str):
            combined_data += chunked_data + "\n\n"
        else:
            combined_data += json.dumps(chunked_data, indent=2) + "\n\n"
    
    # Add NER data
    if ner_data:
        combined_data += "## Named Entity Data\n\n"
        combined_data += ner_data + "\n\n"
    
    # Add ingested data
    if ingested_data:
        combined_data += "## Original Document Content\n\n"
        combined_data += ingested_data + "\n\n"
    
    logger.info(f"Combined {len(combined_data)} characters of data from all sources")
    return combined_data

def save_combined_data(combined_data: str, output_file: str) -> bool:
    """Save combined data to a file"""
    logger.info(f"Saving combined data to {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(combined_data)
        logger.info(f"Successfully saved combined data to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving combined data: {e}")
        return False

def initialize_hirag(config: Dict[str, Any], working_dir: str, 
                     use_neo4j: bool = False, 
                     enable_naive_rag: bool = True,
                     enable_hierarchical_mode: bool = True,
                     neo4j_url: Optional[str] = None,
                     neo4j_user: Optional[str] = None,
                     neo4j_password: Optional[str] = None,
                     chunk_token_size: int = 1200,
                     chunk_overlap_token_size: int = 100,
                     embedding_batch_num: int = 32,
                     embedding_func_max_async: int = 8,
                     max_graph_cluster_size: int = 10,
                     use_hnswlib: bool = False,
                     provider: str = "openai") -> HiRAG:
    """Initialize HiRAG with all available options"""
    logger.info(f"Initializing HiRAG with working directory: {working_dir}")
    logger.info(f"Neo4j integration: {use_neo4j}")
    logger.info(f"Using provider: {provider}")
    
    # Prepare Neo4j authentication if needed
    addon_params = {}
    if use_neo4j:
        # Use environment variables as the default, then override with parameters if provided
        env_neo4j_url = os.environ.get("NEO4J_URI", "neo4j://localhost:7687")
        env_neo4j_user = os.environ.get("NEO4J_USER", "neo4j")
        env_neo4j_password = os.environ.get("NEO4J_PASSWORD", "password")
        
        # Override with config values if present and not provided as parameters
        config_neo4j = config.get('neo4j', {})
        if not neo4j_url and 'url' in config_neo4j:
            neo4j_url = config_neo4j['url']
        if not neo4j_user and 'user' in config_neo4j:
            neo4j_user = config_neo4j['user']
        if not neo4j_password and 'password' in config_neo4j:
            neo4j_password = config_neo4j['password']
        
        # Use parameters if provided, otherwise use environment variables
        final_neo4j_url = neo4j_url or env_neo4j_url
        final_neo4j_user = neo4j_user or env_neo4j_user
        final_neo4j_password = neo4j_password or env_neo4j_password
        
        # Validate the Neo4j URL has a valid scheme
        if not final_neo4j_url or '://' not in final_neo4j_url:
            default_url = "neo4j://localhost:7687"
            logger.warning(f"Invalid Neo4j URI format: '{final_neo4j_url}'. URI must include a scheme (e.g., neo4j:// or bolt://). Using default: {default_url}")
            final_neo4j_url = default_url
        else:
            # Verify the scheme is supported
            scheme = final_neo4j_url.split('://')[0].lower()
            if scheme not in ['bolt', 'neo4j', 'bolt+s', 'neo4j+s', 'bolt+ssc', 'neo4j+ssc']:
                logger.warning(f"Unsupported Neo4j URI scheme: '{scheme}://'. Valid schemes are bolt:// or neo4j://. Using provided scheme anyway.")
        
        logger.info(f"Using Neo4j at {final_neo4j_url} with user {final_neo4j_user}")
        addon_params = {
            "neo4j_url": final_neo4j_url,
            "neo4j_auth": (final_neo4j_user, final_neo4j_password)
        }
    
    # Get embedding dimensions based on provider
    if provider.lower() == "cohere":
        embedding_dim = config.get('model_params', {}).get('cohere_embedding_dim', 1024)
    elif provider.lower() == "deepseek":
        embedding_dim = config.get('model_params', {}).get('deepseek_embedding_dim', 3072)
    elif provider.lower() == "ollama":
        embedding_dim = config.get('model_params', {}).get('ollama_embedding_dim', 3072)
    else:  # openai or azure
        embedding_dim = config.get('model_params', {}).get('openai_embedding_dim', 1536)
    
    # Select the appropriate vector db and graph storage classes
    vector_db_cls = HNSWVectorStorage if use_hnswlib else NanoVectorDBStorage
    graph_storage_cls = Neo4jStorage if use_neo4j else NetworkXStorage
    
    # Initialize HiRAG with all available options
    try:
        graph_func = HiRAG(
            working_dir=working_dir,
            # Graph mode settings
            enable_local=True,
            enable_naive_rag=enable_naive_rag,
            enable_hierachical_mode=enable_hierarchical_mode,
            
            # Text chunking settings
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            tiktoken_model_name="cl100k_base",
            
            # Entity extraction settings
            entity_extract_max_gleaning=1,
            entity_summary_to_max_tokens=500,
            
            # Graph clustering settings
            graph_cluster_algorithm="leiden",
            max_graph_cluster_size=max_graph_cluster_size,
            graph_cluster_seed=0xDEADBEEF,
            
            # Node embedding settings
            node_embedding_algorithm="node2vec",
            node2vec_params={
                "dimensions": embedding_dim,
                "num_walks": 10,
                "walk_length": 40,
                "window_size": 2,
                "iterations": 3,
                "random_seed": 3,
            },
            
            # Storage settings
            key_string_value_json_storage_cls=JsonKVStorage,
            vector_db_storage_cls=vector_db_cls,
            vector_db_storage_cls_kwargs={},
            graph_storage_cls=graph_storage_cls,
            enable_llm_cache=True,
            
            # Additional settings
            always_create_working_dir=True,
            using_azure_openai=(provider.lower() == "azure"),
            addon_params=addon_params,
            
            # Let the HiRAG class determine the appropriate model functions
            # based on the PROVIDER and OPENAI_MODEL_NAME env vars
        )
    except Exception as e:
        logger.error(f"Error initializing HiRAG: {e}")
        if use_neo4j:
            logger.error(f"Neo4j connection parameters: URL={final_neo4j_url}, User={final_neo4j_user}")
            logger.error("Make sure Neo4j is running and the connection details are correct")
        raise
    
    logger.info("HiRAG initialization completed")
    return graph_func

def index_document_for_hirag(graph_func: HiRAG, document_path: str) -> bool:
    """Index a document in HiRAG"""
    logger.info(f"Indexing document for HiRAG: {document_path}")
    
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        graph_func.insert(content)
        logger.info(f"Successfully indexed document: {document_path}")
        return True
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        return False

def process_query(graph_func: HiRAG, query: str, mode: str = "hi") -> str:
    """Process a query using HiRAG"""
    logger.info(f"Processing query: {query} (mode: {mode})")
    
    result = graph_func.query(query, param=QueryParam(mode=mode))
    return result

def main():
    """Main function to integrate the pipeline components with HiRAG"""
    parser = argparse.ArgumentParser(description='HiRAG Pipeline Integration with Full Features')
    parser.add_argument('--ingest-dir', type=str, required=True, help='Directory containing ingested data')
    parser.add_argument('--ner-dir', type=str, required=True, help='Directory containing NER data')
    parser.add_argument('--chunker-dir', type=str, required=True, help='Directory containing chunked data')
    parser.add_argument('--output-dir', type=str, default='', help='Directory to save output (defaults to hi_rag/data)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--query', type=str, help='Query to process')
    parser.add_argument('--mode', type=str, default='hi', 
                        choices=['hi', 'naive', 'hi_nobridge', 'hi_local', 'hi_global', 'hi_bridge'],
                        help='Query mode')
    parser.add_argument('--clean', action='store_true', help='Clean the vector database before starting')
    
    # Neo4j integration options
    parser.add_argument('--use-neo4j', action='store_true', help='Use Neo4j for graph storage')
    parser.add_argument('--neo4j-url', type=str, help='Neo4j URL (default: from env or neo4j://localhost:7687)')
    parser.add_argument('--neo4j-user', type=str, help='Neo4j username (default: from env or neo4j)')
    parser.add_argument('--neo4j-password', type=str, help='Neo4j password (default: from env or password)')
    
    # Advanced HiRAG options
    parser.add_argument('--naive-rag', action='store_true', help='Enable naive RAG mode')
    parser.add_argument('--no-hierarchical', action='store_false', dest='hierarchical', 
                        help='Disable hierarchical mode')
    parser.add_argument('--chunk-size', type=int, default=1200, 
                        help='Chunk token size for text splitting')
    parser.add_argument('--chunk-overlap', type=int, default=100, 
                        help='Chunk overlap token size for text splitting')
    parser.add_argument('--max-cluster-size', type=int, default=10, 
                        help='Maximum graph cluster size')
    parser.add_argument('--use-hnswlib', action='store_true', 
                        help='Use HNSWLib for vector database instead of NanoVectorDB')
    parser.add_argument('--embedding-batch', type=int, default=32, 
                        help='Embedding batch number')
    parser.add_argument('--embedding-async', type=int, default=8, 
                        help='Maximum concurrent embedding function calls')
    
    # Model and provider options
    parser.add_argument('--model', type=str, default='', 
                        help='Override the model to use (sets OPENAI_MODEL_NAME environment variable)')
    parser.add_argument('--provider', type=str, default='', 
                        choices=['openai', 'ollama', 'deepseek', 'azure', 'glm', 'cohere'],
                        help='Model provider: openai, ollama, deepseek, glm, or azure')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set provider from argument or environment variable
    provider = args.provider or os.environ.get('PROVIDER', '')
    
    # Set up provider configuration
    setup_provider_config(config, provider, args.model)
    
    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean vector database if requested
    if args.clean:
        clean_vector_database(output_dir)
    
    # Load data from each pipeline component
    chunked_data = load_chunked_data(args.chunker_dir)
    ner_data = load_ner_data(args.ner_dir)
    ingested_data = load_ingested_data(args.ingest_dir)
    
    # Combine data
    combined_data = combine_data(chunked_data, ner_data, ingested_data)
    
    # Save combined data to a file
    combined_file = os.path.join(output_dir, "combined_pipeline_data.txt")
    save_combined_data(combined_data, combined_file)
    
    # Initialize HiRAG with all options
    try:
        provider = os.environ.get('PROVIDER', 'glm')
        logger.info(f"Initializing HiRAG with provider: {provider}")
        
        graph_func = initialize_hirag(
            config=config,
            working_dir=output_dir,
            use_neo4j=args.use_neo4j,
            enable_naive_rag=args.naive_rag,
            enable_hierarchical_mode=args.hierarchical,
            neo4j_url=args.neo4j_url,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password,
            chunk_token_size=args.chunk_size,
            chunk_overlap_token_size=args.chunk_overlap,
            max_graph_cluster_size=args.max_cluster_size,
            use_hnswlib=args.use_hnswlib,
            embedding_batch_num=args.embedding_batch,
            embedding_func_max_async=args.embedding_async,
            provider=provider
        )
    except Exception as e:
        logger.error(f"Error initializing HiRAG: {e}")
        return 1
    
    # Index the combined document
    success = index_document_for_hirag(graph_func, combined_file)
    if not success:
        logger.error("Failed to index document for HiRAG")
        return 1
    
    # Process query if provided
    if args.query:
        logger.info(f"Processing query: {args.query} (mode: {args.mode})")
        try:
            result = process_query(graph_func, args.query, args.mode)
            print("\nQuery Result:")
            print("-" * 80)
            print(result)
            print("-" * 80)
            
            # Save result to a file
            result_file = os.path.join(output_dir, "query_result.txt")
            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(result)
            logger.info(f"Saved query result to {result_file}")
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return 1
    else:
        logger.info("No query provided. Document has been indexed for later querying.")
        print("\nDocument has been indexed successfully.")
        print(f"You can now run queries using the hi_rag_demo.py or run_hirag.py scripts.")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 