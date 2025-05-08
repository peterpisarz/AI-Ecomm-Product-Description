# AI-Powered Product Description Generator for E-Commerce

This project uses PyTorch and HuggingFace Transformers to generate compelling, SEO-optimized product descriptions with smart product tagging for e-commerce businesses.

## Features

- Generate multiple unique product descriptions from basic product information
- Automatically generate relevant product tags to improve searchability
- Customize tone and style of descriptions (casual, professional, luxury, etc.)
- Include SEO keywords for better search engine ranking
- Bulk generation for multiple products
- Web interface for easy interaction
- Category-specific suggestions and optimization

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone this repository
```bash
git clone <repository-url>
cd ai-ecomm
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your specific settings
```

### Running the Application

#### Development
```bash
python app.py
```

#### Production
```bash
gunicorn -w 4 "app:create_app()"
```

### Deploying to Render

This application is configured for deployment on Render.com as a web service:

1. Push your code to a Git repository (GitHub, GitLab, etc.)
2. Create a new Web Service on Render
3. Connect your Git repository
4. Render will automatically use the render.yaml configuration

Alternatively, you can configure manually:
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn 'simple_app:create_app()' --bind 0.0.0.0:$PORT`

Required environment variables:
- `PORT`: Set by Render automatically
- `DEBUG`: Set to false
- `SECRET_KEY`: Set to a secure random string

#### Testing Locally

To test the Render configuration locally:
```bash
gunicorn 'simple_app:create_app()' --bind 0.0.0.0:5000
```

## Project Structure

- `app.py` - Flask application entry point
- `model/` - PyTorch model definitions and training code
- `data/` - Scripts for data collection and preprocessing
- `utils/` - Helper utilities
  - `seo.py` - SEO analysis and optimization tools
  - `tagging.py` - Smart product tagging functionality
- `web/` - Web interface templates and static files
- `config.py` - Configuration settings

## API Endpoints

- `POST /api/generate` - Generate product descriptions with provided information
- `POST /api/suggest_tags` - Get AI-suggested tags for a product

## Smart Tagging

The application automatically generates relevant product tags using:
- TF-IDF based keyword extraction
- Category-specific tag recommendations
- Multi-word phrase detection
- Inappropriate content filtering

### Sample Usage

```python
from utils.tagging import suggest_product_tags, CATEGORY_TAXONOMY

product_info = {
    'product_name': 'Wireless Bluetooth Earbuds',
    'category': 'electronics',
    'features': 'Noise cancellation | 8-hour battery | Waterproof',
    'description': 'High-quality wireless earbuds with premium sound and comfort.'
}

tags = suggest_product_tags(product_info, CATEGORY_TAXONOMY)
# Returns: ['tech', 'electronic', 'wireless', 'bluetooth', 'noise cancellation']
```

## Development Roadmap

- [x] Initial model training with PyTorch
- [x] Basic web interface
- [x] Smart product tagging system
- [ ] Integration with e-commerce platforms
- [ ] Advanced customization options
- [ ] Performance optimization
- [ ] User management and API access 