#!/usr/bin/env python3
"""
Script to process sample data for the product description generator.
"""
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_sample_data():
    """Process the sample data file and create processed data files."""
    # Paths
    input_file = 'data/raw/sample_products.csv'
    output_dir = 'data/processed'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        logger.info(f"Loaded {len(df)} products from {input_file}")
        
        # Add synthetic tone labels
        df['tone'] = 'neutral'
        
        # Detect luxury tone
        luxury_keywords = ['premium', 'luxury', 'exclusive', 'high-end', 'sophisticated', 
                          'elegant', 'exquisite', 'refined', 'prestigious', 'exceptional']
        luxury_pattern = '|'.join(luxury_keywords)
        df.loc[df['description'].str.contains(luxury_pattern, case=False), 'tone'] = 'luxury'
        
        # Detect technical tone
        technical_keywords = ['specifications', 'technology', 'feature', 'system', 'technical',
                            'performance', 'efficiency', 'precision', 'capability', 'function']
        technical_pattern = '|'.join(technical_keywords)
        df.loc[df['description'].str.contains(technical_pattern, case=False), 'tone'] = 'technical'
        
        # Detect professional tone
        professional_keywords = ['professional', 'business', 'industry', 'standard', 'quality',
                                'reliable', 'effective', 'efficient', 'practical', 'valuable']
        professional_pattern = '|'.join(professional_keywords)
        df.loc[df['description'].str.contains(professional_pattern, case=False), 'tone'] = 'professional'
        
        logger.info(f"Added synthetic tone labels. Distribution: {df['tone'].value_counts().to_dict()}")
        
        # For simplicity with our small dataset, we'll use the same data for train/val/test
        processed_path = os.path.join(output_dir, 'processed_data.csv')
        df.to_csv(processed_path, index=False)
        
        # Create symlinks or copies for train/val/test to maintain compatibility
        train_path = os.path.join(output_dir, 'train.csv')
        val_path = os.path.join(output_dir, 'val.csv')
        test_path = os.path.join(output_dir, 'test.csv')
        
        # Simple copy approach (works everywhere)
        df.to_csv(train_path, index=False)
        df.to_csv(val_path, index=False)
        df.to_csv(test_path, index=False)
        
        logger.info(f"Saved processed data to {output_dir}")
        logger.info(f"Created {len(df)} examples in each of train, validation, and test sets")
        
        return train_path, val_path, test_path
    
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return None, None, None

if __name__ == "__main__":
    process_sample_data() 