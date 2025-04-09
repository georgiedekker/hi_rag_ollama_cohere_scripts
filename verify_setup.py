#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_hirag_installation():
    """Check if HiRAG is installed, and install it if not"""
    print("Checking HiRAG installation...")
    
    try:
        import hirag
        print(f"✅ HiRAG is installed (version: {getattr(hirag, '__version__', 'unknown')})")
        return True
    except ImportError:
        print("❌ HiRAG is not installed. Attempting to install...")
        
        # Try direct installation from GitHub
        print("Installing HiRAG directly from GitHub...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/hhy-huang/HiRAG.git"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error installing HiRAG from GitHub: {result.stderr}")
            
            # Fallback: Clone and install locally
            print("Trying alternative installation method...")
            hirag_repo = Path("../HiRAG")
            if not hirag_repo.exists():
                print("Cloning HiRAG repository...")
                result = subprocess.run(
                    ["git", "clone", "https://github.com/hhy-huang/HiRAG.git", str(hirag_repo)],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Error cloning HiRAG repository: {result.stderr}")
                    return False
            
            # Install HiRAG
            print("Installing HiRAG package...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(hirag_repo)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"Error installing HiRAG: {result.stderr}")
                return False
        
        # Verify installation
        try:
            import hirag
            print(f"✅ HiRAG successfully installed (version: {getattr(hirag, '__version__', 'unknown')})")
            return True
        except ImportError:
            print("❌ Failed to install HiRAG")
            return False

def check_requirements():
    """Check if all requirements are installed"""
    print("Checking requirements...")
    
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        if os.path.exists('../HiRAG/requirements.txt'):
            requirements_file = Path('../HiRAG/requirements.txt')
        else:
            print("❌ requirements.txt not found")
            return False
    
    # Install requirements
    print(f"Installing requirements from {requirements_file}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error installing requirements: {result.stderr}")
        return False
    
    print("✅ Requirements installed")
    return True

def setup_data_directory():
    """Set up data directory for HiRAG"""
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"Creating data directory: {data_dir}")
        data_dir.mkdir(exist_ok=True)
    
    return True

def display_usage_hints():
    """Display usage hints after successful setup"""
    print("\nNow you can use HiRAG with the following commands:")
    print("\n1. Using the shell script (recommended):")
    print("   ./run.sh -q \"What is HiRAG?\" -c")
    print("     -c will clean the vector DB to fix any dimension mismatches")
    
    print("\n2. Using Python directly:")
    print("   python run_hirag.py --query \"What is HiRAG?\" --clean")
    
    print("\nIf you encounter any issues, check the README.md for troubleshooting tips.")

def patch_hirag_installation():
    """Patch the HiRAG installation to fix OpenAI client issue"""
    print("Patching HiRAG installation for Ollama compatibility...")
    
    try:
        import hirag
        hirag_dir = Path(hirag.__file__).parent
        llm_file = hirag_dir / "_llm.py"
        
        if not llm_file.exists():
            print(f"❌ Could not find {llm_file}")
            return False
        
        # Read current file content
        with open(llm_file, 'r') as f:
            content = f.read()
        
        # Check if already patched
        if "os.environ.get(\"OPENAI_API_BASE\"" in content and "gpt_custom_model_complete" in content:
            print("✅ HiRAG is already patched")
            return True
        
        # Create a backup
        backup_file = llm_file.with_suffix('.py.bak')
        with open(backup_file, 'w') as f:
            f.write(content)
        print(f"Created backup at {backup_file}")
        
        # Apply the patch - removing any hardcoded values
        patched_content = content.replace(
            "def get_openai_async_client_instance():\n    global global_openai_async_client\n    if global_openai_async_client is None:\n        global_openai_async_client = AsyncOpenAI()",
            "def get_openai_async_client_instance():\n    global global_openai_async_client\n    if global_openai_async_client is None:\n        # Get configuration from environment variables\n        base_url = os.environ.get(\"OPENAI_API_BASE\", os.environ.get(\"OPENAI_BASE_URL\", None))\n        api_key = os.environ.get(\"OPENAI_API_KEY\", \"\")\n        \n        # Create the client with the environment variables\n        if base_url and api_key:\n            global_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)\n        else:\n            # Fall back to default client, but this will likely fail without proper config\n            global_openai_async_client = AsyncOpenAI()"
        )
        
        patched_content = patched_content.replace(
            "def get_azure_openai_async_client_instance():\n    global global_azure_openai_async_client\n    if global_azure_openai_async_client is None:\n        global_azure_openai_async_client = AsyncAzureOpenAI()",
            "def get_azure_openai_async_client_instance():\n    global global_azure_openai_async_client\n    if global_azure_openai_async_client is None:\n        # Check for environment variables for Azure configuration\n        api_version = os.environ.get(\"AZURE_OPENAI_API_VERSION\", \"2023-05-15\")\n        azure_endpoint = os.environ.get(\"AZURE_OPENAI_ENDPOINT\")\n        azure_key = os.environ.get(\"AZURE_OPENAI_API_KEY\")\n        \n        # If Azure configuration is available, use it\n        if azure_endpoint and azure_key and api_version:\n            global_azure_openai_async_client = AsyncAzureOpenAI(\n                api_version=api_version,\n                azure_endpoint=azure_endpoint,\n                api_key=azure_key\n            )\n        else:\n            # Fall back to the OpenAI client with environment settings\n            base_url = os.environ.get(\"OPENAI_API_BASE\", os.environ.get(\"OPENAI_BASE_URL\"))\n            api_key = os.environ.get(\"OPENAI_API_KEY\")\n            \n            if base_url and api_key:\n                global_azure_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)\n            else:\n                # This will likely fail without proper config\n                global_azure_openai_async_client = AsyncOpenAI()"
        )
        
        # Add custom model functions if not already present
        if "async def gpt_custom_model_complete(" not in patched_content:
            # Find location to insert after gpt_4o_mini_complete
            insertion_point = patched_content.find("async def gpt_4o_mini_complete(")
            end_of_function = patched_content.find(")", insertion_point)
            end_of_function = patched_content.find("}", end_of_function)
            if end_of_function == -1:  # If } not found, try with )
                end_of_function = patched_content.find(")", insertion_point)
            insert_position = patched_content.find("\n", end_of_function) + 1
            
            custom_model_function = """
async def gpt_custom_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Get model name from environment variables or fallback to a default
    model_name = os.environ.get("OPENAI_MODEL_NAME", "llama3")
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
"""
            patched_content = patched_content[:insert_position] + custom_model_function + patched_content[insert_position:]
        
        # Add Azure custom model function
        if "async def azure_openai_custom_model_complete(" not in patched_content:
            # Find location to insert after azure_gpt_4o_mini_complete
            insertion_point = patched_content.find("async def azure_gpt_4o_mini_complete(")
            end_of_function = patched_content.find(")", insertion_point)
            end_of_function = patched_content.find("}", end_of_function)
            if end_of_function == -1:  # If } not found, try with )
                end_of_function = patched_content.find(")", insertion_point)
            insert_position = patched_content.find("\n", end_of_function) + 1
            
            custom_model_function = """
async def azure_openai_custom_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Get model name from environment variables or fallback to a default
    model_name = os.environ.get("OPENAI_MODEL_NAME", "llama3")
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
"""
            patched_content = patched_content[:insert_position] + custom_model_function + patched_content[insert_position:]
        
        # Write the patched content
        with open(llm_file, 'w') as f:
            f.write(patched_content)
        
        # Now check for hirag.py and update it to use the custom model
        hirag_py_file = hirag_dir / "hirag.py"
        if hirag_py_file.exists():
            with open(hirag_py_file, 'r') as f:
                hirag_content = f.read()
            
            # Create a backup
            backup_file = hirag_py_file.with_suffix('.py.bak')
            with open(backup_file, 'w') as f:
                f.write(hirag_content)
            
            # Add the imports for custom model functions
            if "gpt_custom_model_complete" not in hirag_content:
                import_point = hirag_content.find("from ._llm import (")
                end_of_import = hirag_content.find(")", import_point)
                hirag_content = hirag_content[:end_of_import] + ",\n    gpt_custom_model_complete,\n    azure_openai_custom_model_complete" + hirag_content[end_of_import:]
            
            # Update the model_func to use environment variable
            if "best_model_func: callable = gpt_4o_mini_complete" in hirag_content:
                hirag_content = hirag_content.replace(
                    "best_model_func: callable = gpt_4o_mini_complete",
                    "best_model_func: callable = field(\n        default_factory=lambda: (\n            # Use environment variable for model if available, otherwise default to gpt-4o-mini\n            gpt_custom_model_complete \n            if os.environ.get(\"OPENAI_MODEL_NAME\") \n            else gpt_4o_mini_complete\n        )\n    )"
                )
            
            # Update the post_init to handle custom model
            if "if self.best_model_func == gpt_4o_complete:" in hirag_content and "elif self.best_model_func == gpt_custom_model_complete:" not in hirag_content:
                hirag_content = hirag_content.replace(
                    "if self.best_model_func == gpt_4o_complete:",
                    "if self.best_model_func == gpt_4o_complete:"
                )
                hirag_content = hirag_content.replace(
                    "if self.best_model_func == gpt_4o_complete:\n                self.best_model_func = azure_gpt_4o_complete",
                    "if self.best_model_func == gpt_4o_complete:\n                self.best_model_func = azure_gpt_4o_complete\n            elif self.best_model_func == gpt_custom_model_complete:\n                self.best_model_func = azure_openai_custom_model_complete\n            elif self.best_model_func == gpt_4o_mini_complete:\n                self.best_model_func = azure_gpt_4o_mini_complete"
                )
            
            with open(hirag_py_file, 'w') as f:
                f.write(hirag_content)
        
        # Check if Ollama is available and has models we can use
        print("Checking for available Ollama models...")
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                if models:
                    model_names = [model.get("name") for model in models]
                    print(f"Found Ollama models: {', '.join(model_names)}")
                    
                    # Set OPENAI_MODEL_NAME to a model that exists
                    # Prefer llama3 if available, otherwise use the first model
                    preferred_models = ["llama3", "mistral", "llama2", "gemma"]
                    selected_model = None
                    
                    for preferred in preferred_models:
                        for model in model_names:
                            if preferred in model.lower():
                                selected_model = model
                                break
                        if selected_model:
                            break
                    
                    if not selected_model and model_names:
                        selected_model = model_names[0]
                    
                    if selected_model:
                        # Write to .env file
                        env_path = Path(".env")
                        env_content = ""
                        if env_path.exists():
                            with open(env_path, 'r') as f:
                                env_content = f.read()
                        
                        if "OPENAI_MODEL_NAME=" not in env_content:
                            with open(env_path, 'a') as f:
                                f.write(f"\n# Automatically set by verify_setup.py\nOPENAI_MODEL_NAME={selected_model}\n")
                            print(f"✅ Set OPENAI_MODEL_NAME={selected_model} in .env file")
                        else:
                            print("⚠️ OPENAI_MODEL_NAME already set in .env file. Not modifying.")
                else:
                    print("⚠️ No Ollama models found. You may need to pull a model.")
            else:
                print(f"⚠️ Error checking Ollama models: {response.status_code} response")
        except Exception as e:
            print(f"⚠️ Error checking Ollama models: {e}")
        
        print("✅ Successfully patched HiRAG for Ollama compatibility")
        return True
    except Exception as e:
        print(f"❌ Error patching HiRAG: {e}")
        return False

def main():
    """Main verification function"""
    print("=== HI_RAG Setup Verification ===")
    
    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    
    # Check requirements
    req_success = check_requirements()
    
    # Check HiRAG installation
    hirag_success = check_hirag_installation()
    
    # Patch HiRAG for Ollama compatibility
    patch_success = patch_hirag_installation() if hirag_success else False
    
    # Setup data directory
    data_success = setup_data_directory()
    
    # Overall status
    if req_success and hirag_success and patch_success and data_success:
        print("\n✅ Setup verification complete. You're ready to use HI_RAG!")
        display_usage_hints()
        sys.exit(0)
    else:
        print("\n❌ Setup verification failed. Please fix the issues before proceeding.")
        sys.exit(1)
    
if __name__ == "__main__":
    main() 