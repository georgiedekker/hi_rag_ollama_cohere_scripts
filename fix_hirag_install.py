#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import importlib.util
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_hirag_installation():
    """Find the installed HiRAG package directory"""
    try:
        # Try to import hirag to get its location
        spec = importlib.util.find_spec('hirag')
        if spec is None:
            return None
        
        # Get the installation directory
        hirag_path = Path(spec.origin).parent
        logger.info(f"Found HiRAG installation at: {hirag_path}")
        return hirag_path
    except Exception as e:
        logger.error(f"Error finding HiRAG installation: {e}")
        return None

def backup_file(file_path):
    """Create a backup of a file"""
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    shutil.copy2(file_path, backup_path)
    logger.info(f"Created backup at {backup_path}")
    return backup_path

def fix_hirag_installation(hirag_path):
    """Fix the syntax error in hirag.py"""
    # Fix the hirag.py file
    hirag_py = hirag_path / "hirag.py"
    if not hirag_py.exists():
        logger.error(f"Could not find {hirag_py}")
        return False
    
    # Create a backup
    backup_file(hirag_py)
    
    # Read current content
    with open(hirag_py, 'r') as f:
        content = f.read()
    
    # Replace problematic code
    # First, look for import section
    import_section = """import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken

from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    gpt_35_turbo_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete
)"""

    # Handle the syntax error with the comma
    if ",\n    " in content and content.count("from ._llm import (") > 0:
        # Fix the import section
        content = content.replace(import_section, """import asyncio
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from typing import Callable, Dict, List, Optional, Type, Union, cast

import tiktoken

from ._llm import (
    gpt_4o_complete,
    gpt_4o_mini_complete,
    gpt_35_turbo_complete,
    openai_embedding,
    azure_gpt_4o_complete,
    azure_openai_embedding,
    azure_gpt_4o_mini_complete
)""")
    
    # Write the fixed content
    with open(hirag_py, 'w') as f:
        f.write(content)
    
    logger.info(f"Fixed syntax error in {hirag_py}")
    
    # Now fix the _llm.py file to properly handle environment variables
    llm_py = hirag_path / "_llm.py"
    if not llm_py.exists():
        logger.error(f"Could not find {llm_py}")
        return False
    
    # Create a backup
    backup_file(llm_py)
    
    # Read current content
    with open(llm_py, 'r') as f:
        llm_content = f.read()
    
    # Update the client initialization to properly use environment variables
    client_init_pattern = """def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client"""
    
    client_init_replacement = """def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        # Check for environment variables for custom OpenAI configuration
        base_url = os.environ.get("OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", None))
        api_key = os.environ.get("OPENAI_API_KEY", 'ollama')  # Default to ollama if not set
        
        # Create the client with the environment variables if they exist
        if base_url:
            global_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client"""
    
    # Update the openai_complete_if_cache function
    if client_init_pattern in llm_content:
        llm_content = llm_content.replace(client_init_pattern, client_init_replacement)
    
    # Update the model name in the gpt functions
    gpt_pattern = """async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )"""
    
    gpt_replacement = """async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Use a model from env if available, otherwise fallback to GPT-4o-mini
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )"""
    
    # Update the gpt_4o_mini_complete function
    if gpt_pattern in llm_content:
        llm_content = llm_content.replace(gpt_pattern, gpt_replacement)
    
    # Write the updated content
    with open(llm_py, 'w') as f:
        f.write(llm_content)
    
    logger.info(f"Updated environment variable handling in {llm_py}")
    return True

def main():
    hirag_path = find_hirag_installation()
    if not hirag_path:
        logger.error("Could not find HiRAG installation. Make sure it's installed.")
        return 1
    
    success = fix_hirag_installation(hirag_path)
    if not success:
        logger.error("Failed to fix HiRAG installation")
        return 1
    
    logger.info("Successfully fixed HiRAG installation. You may need to restart your application.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 