import os
import logging
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pandas as pd
from model import ProductDescriptionGenerator
from utils.seo import analyze_seo, suggest_improvements
from utils.tagging import suggest_product_tags, CATEGORY_TAXONOMY, filter_inappropriate_tags
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                template_folder='web/templates',
                static_folder='web/static')
    
    # Load configuration
    app.config['SECRET_KEY'] = config.SECRET_KEY
    app.config['DEBUG'] = config.DEBUG
    
    # Initialize the model
    model_path = config.MODEL_PATH
    model_name = config.MODEL_NAME
    
    # Only load the model if it exists, otherwise use the default model
    if os.path.exists(model_path):
        generator = ProductDescriptionGenerator(model_path=model_path)
    else:
        generator = ProductDescriptionGenerator(model_name=model_name)
    
    @app.route('/')
    def index():
        """Render the home page."""
        # Get tone options from config
        tone_options = list(config.TONE_PRESETS.keys())
        # Get category options from taxonomy
        category_options = list(CATEGORY_TAXONOMY.keys())
        return render_template('index.html', tone_options=tone_options, category_options=category_options)
    
    @app.route('/generate', methods=['POST'])
    def generate():
        """Generate product descriptions."""
        try:
            # Get form data
            product_name = request.form.get('product_name')
            category = request.form.get('category')
            features = request.form.get('features', '')
            keywords = request.form.get('keywords', '')
            tone = request.form.get('tone')
            num_descriptions = int(request.form.get('num_descriptions', config.NUM_DESCRIPTIONS))
            
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
            
            # Generate descriptions
            descriptions = generator.generate_descriptions(
                product_info=product_info,
                num_descriptions=num_descriptions,
                tone=tone,
                max_length=config.MAX_LENGTH
            )
            
            # Process keywords for SEO analysis
            keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
            
            # Analyze SEO for each description
            seo_analyses = []
            for desc in descriptions:
                analysis = analyze_seo(desc, keyword_list)
                suggestions = suggest_improvements(analysis, keyword_list)
                seo_analyses.append({
                    'analysis': analysis,
                    'suggestions': suggestions
                })
            
            # Generate product tags for each description
            tags_list = []
            for i, desc in enumerate(descriptions):
                # Add the description to the product info for tag generation
                product_with_desc = product_info.copy()
                product_with_desc['description'] = desc
                
                # Generate and filter tags
                tags = suggest_product_tags(product_with_desc, CATEGORY_TAXONOMY)
                filtered_tags = filter_inappropriate_tags(tags)
                tags_list.append(filtered_tags)
            
            # Render results page
            return render_template(
                'results.html',
                product_info=product_info,
                descriptions=descriptions,
                seo_analyses=seo_analyses,
                tags_list=tags_list,
                tone=tone
            )
            
        except Exception as e:
            logger.error(f"Error generating descriptions: {e}")
            flash(f"An error occurred: {str(e)}", 'error')
            return redirect(url_for('index'))
    
    @app.route('/api/generate', methods=['POST'])
    def api_generate():
        """API endpoint for generating product descriptions."""
        try:
            # Get JSON data
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            
            # Extract product info
            product_name = data.get('product_name')
            category = data.get('category')
            features = data.get('features', '')
            keywords = data.get('keywords', '')
            tone = data.get('tone')
            num_descriptions = int(data.get('num_descriptions', config.NUM_DESCRIPTIONS))
            
            # Validate required fields
            if not product_name or not category:
                return jsonify({'error': 'Product name and category are required'}), 400
            
            # Prepare product info
            product_info = {
                'product_name': product_name,
                'category': category,
                'features': features,
                'keywords': keywords
            }
            
            # Generate descriptions
            descriptions = generator.generate_descriptions(
                product_info=product_info,
                num_descriptions=num_descriptions,
                tone=tone,
                max_length=config.MAX_LENGTH
            )
            
            # Process keywords for SEO analysis
            keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
            
            # Analyze SEO for each description
            seo_analyses = []
            for desc in descriptions:
                analysis = analyze_seo(desc, keyword_list)
                suggestions = suggest_improvements(analysis, keyword_list)
                seo_analyses.append({
                    'analysis': analysis,
                    'suggestions': suggestions
                })
            
            # Generate product tags for each description
            tags_list = []
            for i, desc in enumerate(descriptions):
                # Add the description to the product info for tag generation
                product_with_desc = product_info.copy()
                product_with_desc['description'] = desc
                
                # Generate and filter tags
                tags = suggest_product_tags(product_with_desc, CATEGORY_TAXONOMY)
                filtered_tags = filter_inappropriate_tags(tags)
                tags_list.append(filtered_tags)
            
            # Return JSON response
            return jsonify({
                'product_info': product_info,
                'descriptions': descriptions,
                'seo_analyses': seo_analyses,
                'tags_list': tags_list,
                'tone': tone
            })
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/bulk', methods=['GET', 'POST'])
    def bulk():
        """Bulk generation from CSV file."""
        if request.method == 'POST':
            try:
                # Check if file was uploaded
                if 'file' not in request.files:
                    flash('No file uploaded', 'error')
                    return redirect(request.url)
                
                file = request.files['file']
                
                # Check if filename is empty
                if file.filename == '':
                    flash('No file selected', 'error')
                    return redirect(request.url)
                
                # Check file extension
                if not file.filename.endswith('.csv'):
                    flash('Only CSV files are supported', 'error')
                    return redirect(request.url)
                
                # Read CSV file
                df = pd.read_csv(file)
                
                # Check required columns
                required_columns = ['product_name', 'category']
                if not all(col in df.columns for col in required_columns):
                    flash(f'CSV file must contain columns: {", ".join(required_columns)}', 'error')
                    return redirect(request.url)
                
                # Get parameters
                tone = request.form.get('tone')
                num_descriptions = int(request.form.get('num_descriptions', 1))
                
                # Generate descriptions for each product
                results = []
                for _, row in df.iterrows():
                    product_info = {
                        'product_name': row['product_name'],
                        'category': row['category'],
                        'features': row.get('features', ''),
                        'keywords': row.get('keywords', '')
                    }
                    
                    # Generate descriptions
                    descriptions = generator.generate_descriptions(
                        product_info=product_info,
                        num_descriptions=num_descriptions,
                        tone=tone,
                        max_length=config.MAX_LENGTH
                    )
                    
                    # Add to results
                    for i, desc in enumerate(descriptions):
                        # Generate tags for each description
                        product_with_desc = product_info.copy()
                        product_with_desc['description'] = desc
                        tags = suggest_product_tags(product_with_desc, CATEGORY_TAXONOMY)
                        filtered_tags = filter_inappropriate_tags(tags)
                        
                        results.append({
                            'product_name': product_info['product_name'],
                            'category': product_info['category'],
                            'features': product_info['features'],
                            'keywords': product_info['keywords'],
                            'description': desc,
                            'tags': ', '.join(filtered_tags),
                            'description_number': i + 1
                        })
                
                # Convert results to DataFrame
                results_df = pd.DataFrame(results)
                
                # Generate CSV file
                csv_filename = 'generated_descriptions.csv'
                csv_path = os.path.join('web/static/downloads', csv_filename)
                os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                results_df.to_csv(csv_path, index=False)
                
                # Render results page
                return render_template(
                    'bulk_results.html',
                    results=results,
                    csv_filename=csv_filename
                )
                
            except Exception as e:
                logger.error(f"Error in bulk generation: {e}")
                flash(f"An error occurred: {str(e)}", 'error')
                return redirect(request.url)
        
        # GET request: render form
        tone_options = list(config.TONE_PRESETS.keys())
        category_options = list(CATEGORY_TAXONOMY.keys())
        return render_template('bulk.html', tone_options=tone_options, category_options=category_options)
    
    @app.route('/about')
    def about():
        """About page."""
        return render_template('about.html')
    
    @app.route('/api/suggest_tags', methods=['POST'])
    def api_suggest_tags():
        """API endpoint for suggesting tags for a product."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
                
            # Extract product info
            product_info = {
                'product_name': data.get('product_name', ''),
                'category': data.get('category', ''),
                'features': data.get('features', ''),
                'description': data.get('description', '')
            }
            
            # Validate minimum required data
            if not product_info['product_name'] or not product_info['category']:
                return jsonify({'error': 'Product name and category are required'}), 400
                
            # Generate tags
            tags = suggest_product_tags(product_info, CATEGORY_TAXONOMY)
            filtered_tags = filter_inappropriate_tags(tags)
            
            return jsonify({'tags': filtered_tags})
            
        except Exception as e:
            logger.error(f"API error suggesting tags: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host=config.HOST, port=config.PORT) 