import pandas as pd
import os
import logging
import sys
import argparse
from sklearn.model_selection import train_test_split

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process scraped product data for model training."""
    
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize the data processor.
        
        Args:
            input_dir (str): Directory with raw data files.
            output_dir (str): Directory to save processed data.
        """
        self.input_dir = input_dir or config.RAW_DATA_PATH
        self.output_dir = output_dir or config.PROCESSED_DATA_PATH
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_data(self, filename=None):
        """Load data from CSV file(s).
        
        Args:
            filename (str): Specific file to load, or None to load all files.
            
        Returns:
            pandas.DataFrame: Loaded data.
        """
        if filename:
            # Load specific file
            if not os.path.isabs(filename):
                filename = os.path.join(self.input_dir, filename)
            
            logger.info(f"Loading data from {filename}")
            return pd.read_csv(filename)
        else:
            # Load all CSV files in the input directory
            all_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
            
            if not all_files:
                logger.warning(f"No CSV files found in {self.input_dir}")
                return pd.DataFrame()
            
            # Load and concatenate all files
            dataframes = []
            for file in all_files:
                file_path = os.path.join(self.input_dir, file)
                logger.info(f"Loading data from {file_path}")
                df = pd.read_csv(file_path)
                dataframes.append(df)
            
            return pd.concat(dataframes, ignore_index=True)
    
    def clean_data(self, df):
        """Clean and preprocess the data.
        
        Args:
            df (pandas.DataFrame): Input data.
            
        Returns:
            pandas.DataFrame: Cleaned data.
        """
        logger.info(f"Cleaning data with {len(df)} rows")
        
        # Remove duplicate products
        df = df.drop_duplicates(subset=['product_name', 'category'], keep='first')
        
        # Drop rows with missing essential information
        df = df.dropna(subset=['product_name', 'category', 'description'])
        
        # Clean text fields
        text_columns = ['product_name', 'description', 'features', 'keywords']
        for col in text_columns:
            if col in df.columns:
                # Replace newlines with spaces
                df[col] = df[col].str.replace('\n', ' ', regex=False)
                # Replace multiple spaces with a single space
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
                # Strip whitespace
                df[col] = df[col].str.strip()
        
        # Filter out descriptions that are too short
        df = df[df['description'].str.len() >= 50]
        
        logger.info(f"Data cleaned, {len(df)} rows remaining")
        return df
    
    def add_synthetic_tones(self, df):
        """Add synthetic tone labels for training tone-specific generation.
        
        Args:
            df (pandas.DataFrame): Input data.
            
        Returns:
            pandas.DataFrame: Data with tone labels.
        """
        # Define tone detection criteria (simplistic approach)
        df['tone'] = 'neutral'
        
        # Detect luxury tone
        luxury_keywords = ['premium', 'luxury', 'exclusive', 'high-end', 'sophisticated', 
                          'elegant', 'exquisite', 'refined', 'prestigious', 'exceptional']
        luxury_pattern = '|'.join(luxury_keywords)
        df.loc[df['description'].str.contains(luxury_pattern, case=False), 'tone'] = 'luxury'
        
        # Detect casual tone
        casual_keywords = ['fun', 'cool', 'awesome', 'amazing', 'great', 'handy',
                          'easy', 'simple', 'quick', 'convenient']
        casual_pattern = '|'.join(casual_keywords)
        df.loc[df['description'].str.contains(casual_pattern, case=False), 'tone'] = 'casual'
        
        # Detect technical tone
        technical_keywords = ['specifications', 'technology', 'feature', 'system', 'technical',
                            'performance', 'efficiency', 'precision', 'capability', 'function']
        technical_pattern = '|'.join(technical_keywords)
        df.loc[df['description'].str.contains(technical_pattern, case=False), 'tone'] = 'technical'
        
        # Detect persuasive tone
        persuasive_keywords = ['must-have', 'essential', 'perfect', 'ideal', 'best',
                              'revolutionary', 'innovative', 'leading', 'superior', 'ultimate']
        persuasive_pattern = '|'.join(persuasive_keywords)
        df.loc[df['description'].str.contains(persuasive_pattern, case=False), 'tone'] = 'persuasive'
        
        # Detect professional tone
        professional_keywords = ['professional', 'business', 'industry', 'standard', 'quality',
                                'reliable', 'effective', 'efficient', 'practical', 'valuable']
        professional_pattern = '|'.join(professional_keywords)
        df.loc[df['description'].str.contains(professional_pattern, case=False), 'tone'] = 'professional'
        
        logger.info(f"Added synthetic tone labels. Distribution: {df['tone'].value_counts().to_dict()}")
        return df
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into training, validation, and test sets.
        
        Args:
            df (pandas.DataFrame): Input data.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of data to use for validation.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # First split out test set
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['category']
        )
        
        # Calculate validation size relative to train+val
        val_size_adjusted = val_size / (1 - test_size)
        
        # Split training and validation sets
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, random_state=random_state, 
            stratify=train_val_df['category']
        )
        
        logger.info(f"Data split: {len(train_df)} training, {len(val_df)} validation, {len(test_df)} test samples")
        return train_df, val_df, test_df
    
    def process(self, input_file=None, test_size=0.2, val_size=0.1):
        """Process the data and save train/val/test splits.
        
        Args:
            input_file (str): Specific file to process, or None to process all files.
            test_size (float): Proportion of data to use for testing.
            val_size (float): Proportion of data to use for validation.
            
        Returns:
            tuple: Paths to the saved files (train_path, val_path, test_path)
        """
        # Load data
        df = self.load_data(input_file)
        
        if df.empty:
            logger.error("No data to process")
            return None, None, None
        
        # Clean data
        df = self.clean_data(df)
        
        # Add synthetic tone labels
        df = self.add_synthetic_tones(df)
        
        # Split data
        train_df, val_df, test_df = self.split_data(df, test_size, val_size)
        
        # Save processed data
        train_path = os.path.join(self.output_dir, 'train.csv')
        val_path = os.path.join(self.output_dir, 'val.csv')
        test_path = os.path.join(self.output_dir, 'test.csv')
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved processed data to {self.output_dir}")
        return train_path, val_path, test_path

def main():
    parser = argparse.ArgumentParser(description='Process product data for model training')
    parser.add_argument('--input_file', type=str, default=None,
                        help='Input CSV file (optional, processes all files if not specified)')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Input directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data to use for validation')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DataProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )
    
    # Process data
    processor.process(
        input_file=args.input_file,
        test_size=args.test_size,
        val_size=args.val_size
    )

if __name__ == "__main__":
    main() 