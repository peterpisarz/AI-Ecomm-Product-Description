import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import (
    GPT2Tokenizer, 
    GPT2LMHeadModel, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import os
import sys
import logging

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ProductDescriptionDataset(Dataset):
    """Dataset for product description generation."""
    
    def __init__(self, data_path, tokenizer, max_length=150):
        """Initialize dataset from CSV file.
        
        Args:
            data_path (str): Path to CSV file with product data.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
            max_length (int): Maximum length of token sequences.
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Format the input text
        input_text = f"Product: {row['product_name']}\n"
        input_text += f"Category: {row['category']}\n"
        
        if 'features' in row and not pd.isna(row['features']):
            input_text += f"Features: {row['features']}\n"
            
        if 'keywords' in row and not pd.isna(row['keywords']):
            input_text += f"Keywords: {row['keywords']}\n"
            
        if 'tone' in row and not pd.isna(row['tone']):
            input_text += f"Tone: {row['tone']}\n"
            
        input_text += "Description:"
        
        # If description is available (for training)
        target_text = ""
        if 'description' in row and not pd.isna(row['description']):
            target_text = row['description']
            
        # Tokenize the input and target
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if target_text:
            targets = self.tokenizer.encode_plus(
                target_text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # For training, we prepare labels
            input_ids = inputs['input_ids'].squeeze()
            labels = targets['input_ids'].squeeze()
            
            return {
                'input_ids': input_ids,
                'attention_mask': inputs['attention_mask'].squeeze(),
                'labels': labels
            }
        else:
            # For inference, just return inputs
            return {
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
            }


class ProductDescriptionGenerator:
    """Model for generating product descriptions."""
    
    def __init__(self, model_name=None, model_path=None):
        """Initialize the model.
        
        Args:
            model_name (str): Name of the pretrained model to use.
            model_path (str): Path to the fine-tuned model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        self.model_name = model_name or config.MODEL_NAME
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        
        # Add padding token if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading fine-tuned model from {model_path}")
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
        else:
            logger.info(f"Loading pretrained model {self.model_name}")
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
            
        self.model.to(self.device)
        
    def train(self, train_data_path, val_data_path=None, output_dir='model/finetuned',
              batch_size=8, learning_rate=5e-5, epochs=3):
        """Fine-tune the model on product description data.
        
        Args:
            train_data_path (str): Path to training data.
            val_data_path (str): Path to validation data.
            output_dir (str): Directory to save the fine-tuned model.
            batch_size (int): Batch size for training.
            learning_rate (float): Learning rate.
            epochs (int): Number of training epochs.
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare datasets
        train_dataset = ProductDescriptionDataset(
            train_data_path, self.tokenizer, max_length=config.MAX_LENGTH
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_data_path and os.path.exists(val_data_path):
            val_dataset = ProductDescriptionDataset(
                val_data_path, self.tokenizer, max_length=config.MAX_LENGTH
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else:
            val_loader = None
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Calculate total steps
        total_steps = len(train_loader) * epochs
        
        # Create scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Create progress bar
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                total_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{epochs}, Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
        # Save final model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved fine-tuned model to {output_dir}")
    
    def evaluate(self, val_loader):
        """Evaluate the model on validation data.
        
        Args:
            val_loader (DataLoader): Validation data loader.
            
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
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
        
        # Tokenize the input
        inputs = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate descriptions
        self.model.eval()
        descriptions = []
        
        for _ in range(num_descriptions):
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
            # Decode the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the description part
            description = generated_text.split("Description:", 1)[-1].strip()
            descriptions.append(description)
        
        return descriptions 