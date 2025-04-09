#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import yaml
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_env_variables():
    """Load environment variables from .env file"""
    env_path = Path('.env')
    if not env_path.exists():
        logger.warning(".env file not found, checking for environment variables directly")
    
    load_dotenv()
    
    # Log the loaded variables (redacting sensitive ones)
    env_vars = {
        'PROVIDER': os.environ.get('PROVIDER'),
        'OPENAI_MODEL': os.environ.get('OPENAI_MODEL'),
        'GLM_MODEL': os.environ.get('GLM_MODEL'),
        'EMBEDDING_MODEL': os.environ.get('EMBEDDING_MODEL'),
        'OPENAI_API_BASE': os.environ.get('OPENAI_API_BASE'),
        'OLLAMA_BASE_URL': os.environ.get('OLLAMA_BASE_URL'),
        'NEO4J_URI': os.environ.get('NEO4J_URI'),
        'NEO4J_USER': os.environ.get('NEO4J_USER'),
        'DEEPSEEK_COMPLETION_MODEL': os.environ.get('DEEPSEEK_COMPLETION_MODEL'),
        'DEEPSEEK_REASONING_MODEL': os.environ.get('DEEPSEEK_REASONING_MODEL'),
    }
    
    # Redact API keys and passwords for logging
    if 'OPENAI_API_KEY' in os.environ:
        env_vars['OPENAI_API_KEY'] = '[REDACTED]'
    if 'DEEPSEEK_API_KEY' in os.environ:
        env_vars['DEEPSEEK_API_KEY'] = '[REDACTED]'
    if 'NEO4J_PASSWORD' in os.environ:
        env_vars['NEO4J_PASSWORD'] = '[REDACTED]'
    
    logger.info(f"Loaded environment variables: {env_vars}")
    return True

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        result = subprocess.run(
            ['ollama', 'list'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        models = []
        
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header line
            if line.strip():
                parts = line.split()
                if len(parts) >= 1:
                    model_name = parts[0]
                    models.append(model_name)
        
        logger.info(f"Found Ollama models: {', '.join(models)}")
        return models
    except subprocess.CalledProcessError as e:
        logger.warning(f"Error getting Ollama models: {e}")
        # If ollama command fails, try to use the environment variable
        if 'OPENAI_MODEL' in os.environ:
            model = os.environ.get('OPENAI_MODEL')
            if model.startswith('ollama/'):
                model = model[7:]  # Strip 'ollama/' prefix
            return [model]
        return []
    except Exception as e:
        logger.warning(f"Unexpected error getting Ollama models: {e}")
        return []

def update_config(config_path='config.yaml'):
    """Update config.yaml based on environment variables"""
    # Load existing config if it exists
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Determine default provider
    provider = os.environ.get('PROVIDER')
    if not provider:
        if 'GLM_MODEL' in os.environ:
            provider = 'glm'
        elif 'DEEPSEEK_API_KEY' in os.environ and os.environ.get('DEEPSEEK_API_KEY'):
            provider = 'deepseek'
        elif os.environ.get('OPENAI_API_BASE', '').find('ollama') >= 0 or os.environ.get('OPENAI_API_KEY') == 'ollama':
            provider = 'ollama'
        else:
            provider = 'openai'
    
    # Update default provider
    config['default_provider'] = provider
    logger.info(f"Setting default provider to: {provider}")
    
    # Update Ollama configuration
    if 'ollama' not in config:
        config['ollama'] = {}
    
    ollama_model = os.environ.get('OPENAI_MODEL', '')
    if ollama_model.startswith('ollama/'):
        ollama_model = ollama_model[7:]  # Strip 'ollama/' prefix
    
    config['ollama']['model'] = '${OPENAI_MODEL}'  # Use variable reference
    config['ollama']['base_url'] = '${OLLAMA_BASE_URL}'
    config['ollama']['api_key'] = '${OPENAI_API_KEY}'
    
    # Update GLM configuration
    if 'glm' not in config:
        config['glm'] = {}
    
    glm_model = os.environ.get('GLM_MODEL', '')
    if glm_model.startswith('ollama/'):
        glm_model = glm_model[7:]  # Strip 'ollama/' prefix
    
    config['glm']['model'] = '${GLM_MODEL}'
    config['glm']['api_key'] = '${OPENAI_API_KEY}'
    config['glm']['base_url'] = '${OPENAI_API_BASE}'
    
    # Update DeepSeek configuration
    if 'deepseek' not in config:
        config['deepseek'] = {}
    
    config['deepseek']['model'] = '${DEEPSEEK_COMPLETION_MODEL}'
    config['deepseek']['api_key'] = '${DEEPSEEK_API_KEY}'
    config['deepseek']['base_url'] = 'https://api.deepseek.com'
    
    # Update model parameters
    if 'model_params' not in config:
        config['model_params'] = {}
    
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 3584))
    config['model_params']['ollama_embedding_dim'] = embedding_dim
    config['model_params']['glm_embedding_dim'] = embedding_dim
    config['model_params']['deepseek_embedding_dim'] = 3072
    config['model_params']['openai_embedding_dim'] = 1536
    config['model_params']['max_token_size'] = 8192
    
    # Update Neo4j configuration
    if 'neo4j' not in config:
        config['neo4j'] = {}
    
    config['neo4j']['url'] = '${NEO4J_URI}'
    config['neo4j']['user'] = '${NEO4J_USER}'
    config['neo4j']['password'] = '${NEO4J_PASSWORD}'
    config['neo4j']['database'] = 'neo4j'
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Updated configuration in {config_path}")
    return True

def main():
    """Main function to update the configuration"""
    logger.info("Updating configuration based on environment variables")
    
    # Load environment variables
    load_env_variables()
    
    # Get Ollama models
    ollama_models = get_ollama_models()
    
    # Update config.yaml
    update_config()
    
    logger.info("Configuration updated successfully")
    return 0

if __name__ == "__main__":
    exit(main()) 