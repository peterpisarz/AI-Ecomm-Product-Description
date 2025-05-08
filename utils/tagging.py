import re
import string
import logging
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import os

# Initialize logger
logger = logging.getLogger(__name__)

# Download NLTK data if not already present
try:
    if not os.path.exists(os.path.join(os.path.expanduser("~"), "nltk_data", "corpora", "stopwords")):
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception as e:
    logger.warning(f"Could not load NLTK stopwords: {e}")
    STOPWORDS = set(["a", "an", "the", "and", "or", "but", "if", "because", "as", "what", 
                     "when", "where", "how", "who", "which", "this", "that", "these", "those"])

def preprocess_text(text):
    """Clean and preprocess text for tagging.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_keywords_tfidf(text, n_keywords=10, ngram_range=(1, 2)):
    """Extract key phrases from text using TF-IDF.
    
    Args:
        text (str): Input text to analyze
        n_keywords (int): Number of keywords to extract
        ngram_range (tuple): Range of n-grams to consider
        
    Returns:
        list: Extracted keywords/phrases
    """
    # Preprocess
    text = preprocess_text(text)
    
    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer(
        max_features=2000,
        stop_words=STOPWORDS, 
        ngram_range=ngram_range,
        min_df=1
    )
    
    try:
        # Extract features
        tfidf_matrix = tfidf_vectorizer.fit_transform([text])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        
        # Get feature scores
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Get top keywords
        keywords = [word for word, score in sorted_scores[:n_keywords]]
        return keywords
    except Exception as e:
        logger.error(f"Error extracting TF-IDF keywords: {e}")
        return []

def suggest_product_tags(product_info, category_taxonomy=None, n_tags=5):
    """Generate product tags based on description and metadata.
    
    Args:
        product_info (dict): Product information (name, category, features, description)
        category_taxonomy (dict): Dictionary of category-specific tags
        n_tags (int): Number of tags to suggest
        
    Returns:
        list: Suggested tags
    """
    # Combine all available text
    combined_text = product_info.get('product_name', '') + ' '
    combined_text += product_info.get('category', '') + ' '
    combined_text += product_info.get('features', '') + ' '
    combined_text += product_info.get('description', '')
    
    # Extract keywords
    keywords = extract_keywords_tfidf(combined_text, n_keywords=n_tags*2, ngram_range=(1, 2))
    
    # Add category-specific tags if available
    category = product_info.get('category', '').lower()
    category_tags = []
    
    if category_taxonomy and category in category_taxonomy:
        category_tags = category_taxonomy[category][:n_tags]
    
    # Prioritize category tags but ensure we don't exceed n_tags
    if category_tags:
        remaining_slots = max(0, n_tags - len(category_tags))
        # Remove any category tags from keywords to avoid duplicates
        keywords = [k for k in keywords if k not in category_tags]
        tags = category_tags + keywords[:remaining_slots]
    else:
        tags = keywords[:n_tags]
    
    return tags

def filter_inappropriate_tags(tags, blocklist=None):
    """Filter out inappropriate tags.
    
    Args:
        tags (list): List of tags to filter
        blocklist (set): Set of inappropriate words
        
    Returns:
        list: Filtered tags
    """
    if not blocklist:
        # Default minimal blocklist
        blocklist = {'inappropriate', 'offensive', 'explicit'}
    
    return [tag for tag in tags if not any(blocked in tag for blocked in blocklist)]

# Sample category taxonomy (would be expanded in production)
CATEGORY_TAXONOMY = {
    'electronics': ['gadget', 'tech', 'device', 'digital', 'electronic'],
    'clothing': ['apparel', 'fashion', 'wear', 'clothing', 'garment'],
    'home': ['decor', 'furniture', 'interior', 'home', 'household'],
    'beauty': ['cosmetic', 'skincare', 'makeup', 'beauty', 'treatment'],
    'books': ['reading', 'book', 'literature', 'publication', 'novel']
} 