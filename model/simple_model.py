#!/usr/bin/env python3
"""
Simplified model implementation for product description generation.
This version uses Hugging Face Transformers without the full PyTorch training setup.
"""
import os
import logging
import sys
import time
import random
from transformers import pipeline, set_seed

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleProductDescriptionGenerator:
    """Simple text generation model for product descriptions using Hugging Face pipelines."""
    
    def __init__(self, model_name=None):
        """Initialize the generator with a pre-trained model.
        
        Args:
            model_name (str): Name of the HuggingFace model to use.
        """
        self.model_name = model_name or config.MODEL_NAME
        logger.info(f"Initializing text generation model: {self.model_name}")
        
        try:
            # Initialize text generation pipeline
            self.generator = pipeline('text-generation', model=self.model_name)
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            # Fallback to a template-based approach if model loading fails
            self.generator = None
            logger.warning("Using template-based fallback generation")
    
    def generate_descriptions(self, product_info, num_descriptions=3, tone=None, 
                             max_length=150, temperature=0.8):
        """Generate product descriptions.
        
        Args:
            product_info (dict): Dictionary with product information.
                Required keys: 'product_name', 'category'
                Optional keys: 'features', 'keywords'
            num_descriptions (int): Number of descriptions to generate.
            tone (str): Tone of the description (casual, professional, luxury, etc.)
            max_length (int): Maximum length of generated description.
            temperature (float): Sampling temperature.
            
        Returns:
            list: List of generated descriptions.
        """
        # Format the input text
        input_text = f"Product: {product_info['product_name']}\n"
        input_text += f"Category: {product_info['category']}\n"
        
        if 'features' in product_info and product_info['features']:
            input_text += f"Features: {product_info['features']}\n"
            
        if 'keywords' in product_info and product_info['keywords']:
            input_text += f"Keywords: {product_info['keywords']}\n"
            
        # Add tone instruction if provided
        if tone:
            if tone in config.TONE_PRESETS:
                input_text += f"Tone: {config.TONE_PRESETS[tone]}\n"
            else:
                input_text += f"Tone: {tone}\n"
            
        input_text += "Description:"
        
        # If pipeline is available, use it
        if self.generator is not None:
            try:
                logger.info(f"Generating descriptions with input: {input_text[:50]}...")
                
                # Set random seed for reproducibility but with some variation
                set_seed(random.randint(1, 10000))
                
                descriptions = []
                for i in range(num_descriptions):
                    # Generate text
                    outputs = self.generator(
                        input_text,
                        max_length=len(input_text.split()) + max_length,
                        temperature=temperature,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1,
                        return_full_text=True
                    )
                    
                    # Extract the description part
                    generated_text = outputs[0]['generated_text']
                    description = generated_text.split("Description:", 1)[-1].strip()
                    
                    # Handle cases where generation might be too short
                    if len(description.split()) < 10:
                        description = self._generate_fallback_description(product_info, tone, i)
                    
                    descriptions.append(description)
                    
                    # Small delay between generations for variety
                    if i < num_descriptions - 1:
                        time.sleep(0.2)
                
                return descriptions
            
            except Exception as e:
                logger.error(f"Error generating with model: {e}")
                # Fall back to template generation
                logger.info("Falling back to template generation")
                return self._generate_template_descriptions(product_info, num_descriptions, tone)
        else:
            # Use template-based generation as fallback
            return self._generate_template_descriptions(product_info, num_descriptions, tone)
    
    def _generate_template_descriptions(self, product_info, num_descriptions, tone):
        """Generate descriptions using templates when ML model is not available.
        
        Args:
            product_info (dict): Product information.
            num_descriptions (int): Number of descriptions to generate.
            tone (str): Tone of the description.
            
        Returns:
            list: List of generated descriptions.
        """
        descriptions = []
        product_name = product_info['product_name']
        category = product_info['category']
        features = product_info.get('features', '')
        
        # Template patterns for different tones
        templates = {
            'casual': [
                "Hey there! Check out our {product} - it's a fantastic {category} that you'll love. {features_text}Perfect for everyday use!",
                "Looking for a great {category}? Our {product} is just what you need! {features_text}Super easy to use and totally worth it.",
                "Need a reliable {category}? Our {product} is awesome! {features_text}You'll wonder how you ever lived without it!"
            ],
            'professional': [
                "Introducing the {product}, a premium {category} designed for optimal performance. {features_text}Ideal for professional environments.",
                "The {product} represents the pinnacle of {category} design and functionality. {features_text}Engineered to exceed professional standards.",
                "Our {product} offers unparalleled quality in the {category} market. {features_text}Trusted by professionals worldwide."
            ],
            'luxury': [
                "Experience the exquisite craftsmanship of our {product}, the most refined {category} available today. {features_text}For those who demand nothing but the best.",
                "The {product} embodies luxury in every aspect of its design. {features_text}Elevate your experience with this exceptional {category}.",
                "Indulge in the opulence of our {product}, a truly remarkable {category}. {features_text}An investment in unparalleled quality and prestige."
            ],
            'technical': [
                "The {product} incorporates cutting-edge technology for superior {category} performance. {features_text}Engineered with precision for optimal results.",
                "Our {product} features advanced specifications that set a new standard for {category} capabilities. {features_text}Designed for technical excellence.",
                "The {product} utilizes innovative systems to deliver exceptional {category} functionality. {features_text}Technical superiority in every component."
            ],
            'persuasive': [
                "Don't miss out on our incredible {product} - the {category} that will transform your experience! {features_text}Order now and see the difference!",
                "Why settle for less when you can have our amazing {product}? This {category} will exceed all your expectations! {features_text}Limited stock available!",
                "Act now to secure your own {product} - the ultimate {category} solution you've been waiting for! {features_text}Satisfaction guaranteed!"
            ]
        }
        
        # Default tone if not specified or not found
        selected_tone = tone if tone in templates else 'professional'
        
        # Process features
        if features:
            feature_list = features.split('|')
            features_formatted = ', '.join(f.strip() for f in feature_list)
            features_text = f"Key features include {features_formatted}. "
        else:
            features_text = ""
        
        # Generate the required number of descriptions
        for i in range(num_descriptions):
            # Cycle through templates if we need more than are available
            template_idx = i % len(templates[selected_tone])
            template = templates[selected_tone][template_idx]
            
            # Format the template
            description = template.format(
                product=product_name,
                category=category,
                features_text=features_text
            )
            
            descriptions.append(description)
        
        return descriptions
    
    def _generate_fallback_description(self, product_info, tone, index):
        """Generate a single fallback description when model output is insufficient.
        
        Args:
            product_info (dict): Product information.
            tone (str): Tone of the description.
            index (int): Index for variety in templates.
            
        Returns:
            str: A fallback description.
        """
        fallback_templates = [
            "This {product} is a high-quality {category} solution designed to meet your needs.",
            "Our {product} offers the best features you'd expect from a premium {category}.",
            "Discover the amazing {product}, the perfect {category} for all your requirements."
        ]
        
        template_idx = index % len(fallback_templates)
        return fallback_templates[template_idx].format(
            product=product_info['product_name'],
            category=product_info['category']
        ) 