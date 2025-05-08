import os
import sys
import logging
import argparse
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

def main():
    """Train the product description generator model."""
    parser = argparse.ArgumentParser(description='Train product description generator model')
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model name to use (default from config)')
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to training data CSV (default from config)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data CSV (default from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the trained model (default from config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training (default from config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate for training (default from config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (default from config)')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum sequence length (default from config)')
    
    args = parser.parse_args()
    
    # Get parameters from args or config
    model_name = args.model_name or config.MODEL_NAME
    train_data_path = args.train_data or config.TRAIN_DATA_PATH
    val_data_path = args.val_data or config.VAL_DATA_PATH
    output_dir = args.output_dir or config.MODEL_PATH
    batch_size = args.batch_size or config.BATCH_SIZE
    learning_rate = args.learning_rate or config.LEARNING_RATE
    epochs = args.epochs or config.NUM_EPOCHS
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    
    # Display training configuration
    logger.info("Training Configuration:")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Training data: {train_data_path}")
    logger.info(f"Validation data: {val_data_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Epochs: {epochs}")
    
    # Create and train the model
    logger.info("Initializing the model...")
    generator = ProductDescriptionGenerator(model_name=model_name)
    
    logger.info("Starting training...")
    generator.train(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs
    )
    
    logger.info(f"Training complete. Model saved to {output_dir}")

if __name__ == "__main__":
    main() 