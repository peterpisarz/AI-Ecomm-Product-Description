import os
import sys
import logging
import argparse
import json
from model import ProductDescriptionGenerator

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_descriptions(product_info, model_path=None, model_name=None, num_descriptions=3, 
                        tone=None, max_length=150, temperature=0.8):
    """Generate product descriptions using the trained model.
    
    Args:
        product_info (dict): Dictionary with product information.
        model_path (str): Path to the fine-tuned model.
        model_name (str): Name of the pretrained model.
        num_descriptions (int): Number of descriptions to generate.
        tone (str): Tone of the description.
        max_length (int): Maximum length of generated description.
        temperature (float): Sampling temperature.
        
    Returns:
        list: List of generated descriptions.
    """
    # Initialize the model
    generator = ProductDescriptionGenerator(
        model_name=model_name or config.MODEL_NAME,
        model_path=model_path or config.MODEL_PATH
    )
    
    # Generate descriptions
    descriptions = generator.generate_descriptions(
        product_info=product_info,
        num_descriptions=num_descriptions,
        tone=tone,
        max_length=max_length,
        temperature=temperature
    )
    
    return descriptions

def main():
    """Generate product descriptions from command line."""
    parser = argparse.ArgumentParser(description='Generate product descriptions')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the fine-tuned model (default from config)')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model name to use (default from config)')
    parser.add_argument('--product_name', type=str, required=True,
                        help='Name of the product')
    parser.add_argument('--category', type=str, required=True,
                        help='Category of the product')
    parser.add_argument('--features', type=str, default='',
                        help='Features of the product (pipe-separated)')
    parser.add_argument('--keywords', type=str, default='',
                        help='Keywords for SEO (comma-separated)')
    parser.add_argument('--num_descriptions', type=int, default=None,
                        help='Number of descriptions to generate (default from config)')
    parser.add_argument('--tone', type=str, default=None,
                        help='Tone of description (casual, professional, luxury, etc.)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum length of description (default from config)')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (higher is more creative, lower is more focused)')
    parser.add_argument('--input_json', type=str, default=None,
                        help='Path to JSON file with product information')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Path to save generated descriptions as JSON')
    
    args = parser.parse_args()
    
    # Get parameters from args or config
    model_path = args.model_path or config.MODEL_PATH
    model_name = args.model_name or config.MODEL_NAME
    num_descriptions = args.num_descriptions or config.NUM_DESCRIPTIONS
    max_length = args.max_length or config.MAX_LENGTH
    
    # Either load from JSON or use command line arguments
    if args.input_json:
        try:
            with open(args.input_json, 'r') as f:
                product_info = json.load(f)
        except Exception as e:
            logger.error(f"Error loading input JSON: {e}")
            return
    else:
        # Create product info dictionary from command line arguments
        product_info = {
            'product_name': args.product_name,
            'category': args.category,
            'features': args.features,
            'keywords': args.keywords
        }
    
    # Check for required fields
    if 'product_name' not in product_info or 'category' not in product_info:
        logger.error("Product name and category are required")
        return
    
    logger.info(f"Generating {num_descriptions} descriptions for {product_info['product_name']}")
    
    # Generate descriptions
    descriptions = generate_descriptions(
        product_info=product_info,
        model_path=model_path,
        model_name=model_name,
        num_descriptions=num_descriptions,
        tone=args.tone,
        max_length=max_length,
        temperature=args.temperature
    )
    
    # Add the descriptions to the product info
    product_info['descriptions'] = descriptions
    
    # Output the results
    if args.output_json:
        try:
            with open(args.output_json, 'w') as f:
                json.dump(product_info, f, indent=2)
            logger.info(f"Saved descriptions to {args.output_json}")
        except Exception as e:
            logger.error(f"Error saving output JSON: {e}")
    else:
        # Print the descriptions to the console
        print(f"\nGenerated {len(descriptions)} descriptions for {product_info['product_name']}:")
        for i, desc in enumerate(descriptions, 1):
            print(f"\nDescription {i}:\n{desc}")

if __name__ == "__main__":
    main() 