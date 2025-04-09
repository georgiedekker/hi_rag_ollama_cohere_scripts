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