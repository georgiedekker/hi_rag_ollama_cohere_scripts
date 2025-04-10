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
    Safely load JSON string, with error handling and repair capabilities.
    
    Args:
        json_str: JSON string to parse
        
    Returns:
        Parsed JSON as Python dict/list
        
    Raises:
        ValueError: If JSON cannot be parsed after all repair attempts
    """
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try to fix common JSON errors
        logger.warning(f"JSON decode error: {str(e)}")
        
        fixed_json = json_str
        error_msg = str(e)
        
        # Fix for missing commas
        if "Expecting ',' delimiter" in error_msg:
            position = e.pos
            # Insert comma at the position
            fixed_json = json_str[:position] + "," + json_str[position:]
            try:
                return json.loads(fixed_json)
            except:
                logger.debug(f"Failed to fix JSON by adding comma at position {position}")
        
        # Fix for unquoted property names
        if "Expecting property name enclosed in double quotes" in error_msg:
            try:
                # Use regex to find and fix unquoted property names
                import re
                # Pattern matches: {word: or ,word: (unquoted property)
                pattern = r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:'
                replacement = r'\1"\2":'
                fixed_json = re.sub(pattern, replacement, json_str)
                
                if fixed_json != json_str:  # Only try if changes were made
                    logger.debug(f"Attempting to fix unquoted property names")
                    return json.loads(fixed_json)
            except Exception as quote_fix_error:
                logger.debug(f"Failed to fix unquoted property names: {quote_fix_error}")
        
        # Try more aggressive repair for improperly nested braces
        if "Extra data" in error_msg or "Expecting value" in error_msg or "Unterminated string" in error_msg:
            try:
                # Ensure properly balanced braces and quotes
                stack = []
                in_string = False
                escape_next = False
                fixed_json = ""
                
                for i, char in enumerate(json_str):
                    if escape_next:
                        escape_next = False
                        fixed_json += char
                        continue
                        
                    if char == '\\':
                        escape_next = True
                        fixed_json += char
                        continue
                        
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        
                    if not in_string:
                        if char == '{' or char == '[':
                            stack.append(char)
                        elif char == '}':
                            if not stack or stack[-1] != '{':
                                continue  # Skip this char as it's unbalanced
                            stack.pop()
                        elif char == ']':
                            if not stack or stack[-1] != '[':
                                continue  # Skip this char as it's unbalanced
                            stack.pop()
                    
                    fixed_json += char
                
                # Close any unclosed brackets/braces
                while stack:
                    closing = '}' if stack[-1] == '{' else ']'
                    fixed_json += closing
                    stack.pop()
                
                logger.debug(f"Attempting more aggressive JSON repair")
                return json.loads(fixed_json)
            except Exception as repair_error:
                logger.debug(f"Aggressive JSON repair failed: {repair_error}")
        
        # Ultimate fallback: try to extract well-formed JSON objects using regex
        try:
            logger.debug("Attempting to extract JSON using regex pattern matching")
            # Look for patterns that might be valid JSON objects
            import re
            # Find the outermost JSON object (simplified approach)
            open_brace = json_str.find('{')
            if open_brace >= 0:
                # Start from the first opening brace
                stack = 1  # We start with one opening brace
                for i in range(open_brace + 1, len(json_str)):
                    if json_str[i] == '{':
                        stack += 1
                    elif json_str[i] == '}':
                        stack -= 1
                        if stack == 0:  # We've found the matching closing brace
                            potential_json = json_str[open_brace:i+1]
                            try:
                                return json.loads(potential_json)
                            except Exception as inner_error:
                                logger.debug(f"Extracted JSON object failed to parse: {inner_error}")
                                # Continue searching from this position
                                open_brace = json_str.find('{', i+1)
                                if open_brace >= 0:
                                    i = open_brace
                                    stack = 1
                                else:
                                    break
            
            # If no valid JSON objects found with the stack method, try a simpler approach
            # Just look for text between { and } (may catch invalid nested structures)
            simple_pattern = r'{[^{}]*}'
            matches = re.findall(simple_pattern, json_str)
            
            if matches:
                # Try each potential JSON object
                for potential_json in matches:
                    try:
                        return json.loads(potential_json)
                    except:
                        continue
        except Exception as regex_error:
            logger.debug(f"JSON extraction fallback failed: {regex_error}")
            
        # If we get here, all repair attempts failed
        raise ValueError(f"Could not parse JSON: {error_msg}") 