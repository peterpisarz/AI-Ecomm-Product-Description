#!/usr/bin/env python3
"""
A simplified Flask application for the product description generator.
This version doesn't require PyTorch or Transformers, just for demo purposes.
"""
import os
import logging
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flask settings
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')
SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

# Tone presets
TONE_PRESETS = {
    'casual': 'Generate a casual, friendly product description with conversational language.',
    'professional': 'Generate a professional, business-oriented product description with formal language.',
    'luxury': 'Generate a premium, high-end product description emphasizing quality and exclusivity.',
    'technical': 'Generate a detailed technical product description focusing on specifications and features.',
    'persuasive': 'Generate a persuasive product description emphasizing benefits and call-to-action.'
}

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='web/templates',
                static_folder='web/static')
    
    # Load configuration
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['DEBUG'] = DEBUG
    
    # Load sample product descriptions for demo
    try:
        sample_data = pd.read_csv('data/processed/processed_data.csv')
    except Exception as e:
        logger.error(f"Error loading sample data: {e}")
        sample_data = pd.DataFrame()
    
    @app.route('/')
    def index():
        """Render the home page."""
        # Get tone options from config
        tone_options = list(TONE_PRESETS.keys())
        return render_template('index.html', tone_options=tone_options)
    
    @app.route('/generate', methods=['POST'])
    def generate():
        """Generate product descriptions (demo version)."""
        try:
            # Get form data
            product_name = request.form.get('product_name')
            category = request.form.get('category')
            features = request.form.get('features', '')
            keywords = request.form.get('keywords', '')
            tone = request.form.get('tone')
            num_descriptions = int(request.form.get('num_descriptions', 3))
            
            # Validate required fields
            if not product_name or not category:
                flash('Product name and category are required.', 'error')
                return redirect(url_for('index'))
            
            # Prepare product info
            product_info = {
                'product_name': product_name,
                'category': category,
                'features': features,
                'keywords': keywords
            }
            
            # For demo, instead of generating with ML, we'll use template-based generation
            descriptions = []
            for i in range(num_descriptions):
                # Start with a template based on tone
                if tone and tone in TONE_PRESETS:
                    template = f"[{tone.upper()}] "
                else:
                    template = ""
                
                # Add product info
                template += f"Introducing the {product_name}, a premium {category} solution. "
                
                # Add features if available
                if features:
                    feature_list = features.split('|')
                    template += f"Key features include: {', '.join(feature.strip() for feature in feature_list)}. "
                
                # Add a closing statement
                template += f"The {product_name} is perfect for anyone looking for a high-quality {category}."
                
                # For variety, add a numbered suffix
                template += f" (Demo description {i+1})"
                
                descriptions.append(template)
            
            # Simple SEO analysis just for the demo
            keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
            seo_analyses = []
            for desc in descriptions:
                seo_score = 75.0 + (len(desc) / 100)  # Just a dummy score
                
                # Count keyword occurrences
                keyword_counts = {}
                for keyword in keyword_list:
                    count = desc.lower().count(keyword.lower())
                    keyword_counts[keyword] = count
                
                analysis = {
                    'seo_score': seo_score,
                    'keyword_counts': keyword_counts,
                    'keyword_density': 0.05,  # Just a dummy value
                    'total_words': len(desc.split()),
                    'keywords_found': sum(1 for k, v in keyword_counts.items() if v > 0),
                    'length_score': 0.8
                }
                
                suggestions = []
                if analysis['total_words'] < 50:
                    suggestions.append("Description is too short. Aim for at least 50 words.")
                
                seo_analyses.append({
                    'analysis': analysis,
                    'suggestions': suggestions
                })
            
            # Render results page
            return render_template(
                'results.html',
                product_info=product_info,
                descriptions=descriptions,
                seo_analyses=seo_analyses,
                tone=tone
            )
            
        except Exception as e:
            logger.error(f"Error generating descriptions: {e}")
            flash(f"An error occurred: {str(e)}", 'error')
            return redirect(url_for('index'))
    
    @app.route('/bulk', methods=['GET', 'POST'])
    def bulk():
        """Bulk generation from CSV file (simplified for demo)."""
        if request.method == 'POST':
            flash('Bulk generation is not implemented in the demo version.', 'info')
            return redirect(url_for('index'))
        
        # GET request: render form
        tone_options = list(TONE_PRESETS.keys())
        return render_template('bulk.html', tone_options=tone_options)
    
    @app.route('/about')
    def about():
        """About page."""
        return render_template('about.html')
    
    # Add a custom filter for nl2br (newline to <br>)
    @app.template_filter('nl2br')
    def nl2br(value):
        if not value:
            return value
        return value.replace('\n', '<br>')
    
    # Add current year for footer
    @app.context_processor
    def inject_now():
        from datetime import datetime
        return {'now': datetime.now()}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host=HOST, port=PORT) 