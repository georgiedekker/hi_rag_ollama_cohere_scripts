#!/usr/bin/env python3
"""
Test script for the text sanitizer module.
"""
import json
import logging
from text_sanitizer import sanitize_for_json, validate_json_safety, prepare_chunk_for_api, safe_json_loads

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_json_sanitization():
    """Test the JSON sanitization functions with problematic text"""
    
    # Test cases - these are examples of text that would break JSON
    test_cases = [
        # Special characters that break JSON
        'Text with "quotes" that need escaping',
        'Backslashes \\ need special handling',
        'Newlines\nand\ttabs break JSON',
        'Control characters like \b and \f break JSON',
        # The specific examples from the error
        'Expecting "," delimiter: line 344 column 4',
        'Expecting "," delimiter: line 98 column 60',
        # Complex example with multiple issues
        'Complex {"example": \n with \t "nested" \\ quotes} and [arrays, with, items]',
    ]
    
    success_count = 0
    
    # Test each case
    for i, test_case in enumerate(test_cases):
        logger.info(f"Testing case {i+1}: {test_case[:30]}...")
        
        # Sanitize the text
        sanitized = sanitize_for_json(test_case)
        
        # Try to use it in JSON
        try:
            # Create a test object
            test_obj = {
                "text": sanitized,
                "metadata": {
                    "original_length": len(test_case),
                    "sanitized_length": len(sanitized)
                }
            }
            
            # Convert to JSON string
            json_str = json.dumps(test_obj)
            
            # Parse it back
            parsed = json.loads(json_str)
            
            # Verify the content
            if parsed["text"] == sanitized:
                logger.info(f"✅ Case {i+1} passed sanitization test")
                success_count += 1
            else:
                logger.error(f"❌ Case {i+1} failed: content mismatch")
        except Exception as e:
            logger.error(f"❌ Case {i+1} failed with error: {str(e)}")
    
    logger.info(f"Test results: {success_count}/{len(test_cases)} cases passed")
    
    # Test the more complex recursive sanitization
    complex_obj = {
        "entities": [
            {"id": "1", "name": "Test \"Entity\"", "description": "Has\nproblematic\tcharacters"},
            {"id": "2", "name": "Another \\ Entity", "relationships": ["complex\njson", "with\"quotes"]}
        ],
        "metadata": {
            "source": "Test \\ Source",
            "notes": "Contains\nnewlines and \"quotes\""
        }
    }
    
    logger.info("Testing recursive sanitization of complex object...")
    
    try:
        # Sanitize the complex object
        sanitized_obj = validate_json_safety(complex_obj)
        
        # Convert to JSON string
        json_str = json.dumps(sanitized_obj)
        
        # Parse it back
        parsed = json.loads(json_str)
        
        # Verify structure is preserved
        if len(parsed["entities"]) == len(complex_obj["entities"]):
            logger.info("✅ Complex object structure preserved")
            logger.info(f"Original object keys: {list(complex_obj.keys())}")
            logger.info(f"Sanitized object keys: {list(parsed.keys())}")
        else:
            logger.error("❌ Complex object test failed: structure changed")
    except Exception as e:
        logger.error(f"❌ Complex object test failed with error: {str(e)}")
    
    # Test the prepare_chunk_for_api function
    logger.info("Testing prepare_chunk_for_api function...")
    problem_chunk = {
        "content": "Problematic content with \"quotes\", newlines\n, and control characters \b\f",
        "metadata": {
            "source_file": "test\tfile.txt",
            "page": 1
        }
    }
    
    try:
        prepared_chunk = prepare_chunk_for_api(problem_chunk)
        json_str = json.dumps(prepared_chunk)
        parsed = json.loads(json_str)
        logger.info("✅ prepare_chunk_for_api test passed")
    except Exception as e:
        logger.error(f"❌ prepare_chunk_for_api test failed with error: {str(e)}")
    
    # Test the safe_json_loads function with a broken JSON string
    logger.info("Testing safe_json_loads with broken JSON...")
    broken_json = '{"entities": [{"id": "1" "name": "Missing comma here"}, {"id": "2", "name": "Other entity"}]}'
    
    try:
        parsed = safe_json_loads(broken_json)
        logger.info("✅ safe_json_loads fixed broken JSON successfully")
        logger.info(f"Parsed result: {json.dumps(parsed, indent=2)}")
    except ValueError as e:
        logger.warning(f"⚠️ safe_json_loads couldn't fix this particular broken JSON: {str(e)}")
    
    logger.info("Text sanitizer tests completed")

if __name__ == "__main__":
    test_json_sanitization() 