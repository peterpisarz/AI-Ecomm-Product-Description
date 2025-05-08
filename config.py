import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Flask settings
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Model settings
MODEL_NAME = os.getenv('MODEL_NAME', 'distilgpt2')
MODEL_PATH = os.getenv('MODEL_PATH', 'model/finetuned')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 150))
NUM_DESCRIPTIONS = int(os.getenv('NUM_DESCRIPTIONS', 3))

# Training settings
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 8))
LEARNING_RATE = float(os.getenv('LEARNING_RATE', 5e-5))
NUM_EPOCHS = int(os.getenv('NUM_EPOCHS', 3))
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', 'data/processed/train.csv')
VAL_DATA_PATH = os.getenv('VAL_DATA_PATH', 'data/processed/val.csv')

# Data settings
RAW_DATA_PATH = os.getenv('RAW_DATA_PATH', 'data/raw')
PROCESSED_DATA_PATH = os.getenv('PROCESSED_DATA_PATH', 'data/processed')

# Tone presets
TONE_PRESETS = {
    'casual': 'Generate a casual, friendly product description with conversational language.',
    'professional': 'Generate a professional, business-oriented product description with formal language.',
    'luxury': 'Generate a premium, high-end product description emphasizing quality and exclusivity.',
    'technical': 'Generate a detailed technical product description focusing on specifications and features.',
    'persuasive': 'Generate a persuasive product description emphasizing benefits and call-to-action.'
} 