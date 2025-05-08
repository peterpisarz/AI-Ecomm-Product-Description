import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import random
import logging
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductScraper:
    """Scraper for e-commerce product data."""
    
    def __init__(self, output_dir=None):
        """Initialize the scraper.
        
        Args:
            output_dir (str): Directory to save scraped data.
        """
        self.output_dir = output_dir or config.RAW_DATA_PATH
        os.makedirs(self.output_dir, exist_ok=True)
        
        # User agent for requests to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        # For storing scraped data
        self.products = []
    
    def scrape_amazon(self, categories, max_products_per_category=100, delay_range=(1, 3)):
        """Scrape product data from Amazon.
        
        Args:
            categories (list): List of category names to scrape.
            max_products_per_category (int): Maximum number of products to scrape per category.
            delay_range (tuple): Range of seconds to wait between requests.
        """
        logger.info(f"Starting Amazon scraping for {len(categories)} categories")
        
        for category in categories:
            logger.info(f"Scraping category: {category}")
            
            # Prepare search URL
            search_url = f"https://www.amazon.com/s?k={category.replace(' ', '+')}"
            
            try:
                # Get search results page
                response = requests.get(search_url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product links
                product_links = []
                for item in soup.select('.s-result-item .a-link-normal.a-text-normal'):
                    href = item.get('href')
                    if href and '/dp/' in href:
                        if href.startswith('/'):
                            product_links.append(f"https://www.amazon.com{href}")
                        else:
                            product_links.append(href)
                
                logger.info(f"Found {len(product_links)} product links for {category}")
                
                # Limit number of products
                product_links = product_links[:max_products_per_category]
                
                # Scrape product details in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(self.scrape_amazon_product, url, category) 
                        for url in product_links
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            product = future.result()
                            if product:
                                self.products.append(product)
                                logger.info(f"Total products scraped: {len(self.products)}")
                        except Exception as e:
                            logger.error(f"Error processing product: {e}")
                
            except Exception as e:
                logger.error(f"Error scraping category {category}: {e}")
            
            # Save intermediate results
            self.save_to_csv()
            
            # Delay to avoid being blocked
            time.sleep(random.uniform(*delay_range))
    
    def scrape_amazon_product(self, url, category):
        """Scrape a single Amazon product.
        
        Args:
            url (str): Product URL.
            category (str): Product category.
            
        Returns:
            dict: Product data.
        """
        try:
            # Delay to avoid being blocked
            time.sleep(random.uniform(0.5, 2))
            
            # Get product page
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract product name
            product_name_elem = soup.select_one('#productTitle')
            if not product_name_elem:
                logger.warning(f"No product title found for {url}")
                return None
                
            product_name = product_name_elem.text.strip()
            
            # Extract product description
            description = ""
            description_elem = soup.select_one('#productDescription')
            if description_elem:
                description = description_elem.text.strip()
            else:
                # Try alternative description locations
                description_elems = soup.select('#feature-bullets .a-list-item')
                if description_elems:
                    description = ' '.join([elem.text.strip() for elem in description_elems])
            
            if not description:
                logger.warning(f"No description found for {product_name}")
                return None
            
            # Extract features
            features = []
            feature_elems = soup.select('#feature-bullets .a-list-item')
            for elem in feature_elems:
                features.append(elem.text.strip())
            
            # Extract keywords (from the meta tags)
            keywords = ""
            keywords_elem = soup.select_one('meta[name="keywords"]')
            if keywords_elem:
                keywords = keywords_elem.get('content', '')
            
            # Create product data dictionary
            product_data = {
                'product_name': product_name,
                'category': category,
                'description': description,
                'features': ' | '.join(features),
                'keywords': keywords,
                'url': url
            }
            
            logger.info(f"Scraped product: {product_name}")
            return product_data
            
        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None
    
    def scrape_etsy(self, categories, max_products_per_category=100, delay_range=(1, 3)):
        """Scrape product data from Etsy.
        
        Args:
            categories (list): List of category names to scrape.
            max_products_per_category (int): Maximum number of products to scrape per category.
            delay_range (tuple): Range of seconds to wait between requests.
        """
        logger.info(f"Starting Etsy scraping for {len(categories)} categories")
        
        for category in categories:
            logger.info(f"Scraping category: {category}")
            
            # Prepare search URL
            search_url = f"https://www.etsy.com/search?q={category.replace(' ', '+')}"
            
            try:
                # Get search results page
                response = requests.get(search_url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find product links
                product_links = []
                for item in soup.select('.listing-link'):
                    href = item.get('href')
                    if href and '/listing/' in href:
                        product_links.append(href)
                
                logger.info(f"Found {len(product_links)} product links for {category}")
                
                # Limit number of products
                product_links = product_links[:max_products_per_category]
                
                # Scrape product details in parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [
                        executor.submit(self.scrape_etsy_product, url, category) 
                        for url in product_links
                    ]
                    
                    for future in as_completed(futures):
                        try:
                            product = future.result()
                            if product:
                                self.products.append(product)
                                logger.info(f"Total products scraped: {len(self.products)}")
                        except Exception as e:
                            logger.error(f"Error processing product: {e}")
                
            except Exception as e:
                logger.error(f"Error scraping category {category}: {e}")
            
            # Save intermediate results
            self.save_to_csv()
            
            # Delay to avoid being blocked
            time.sleep(random.uniform(*delay_range))
    
    def scrape_etsy_product(self, url, category):
        """Scrape a single Etsy product.
        
        Args:
            url (str): Product URL.
            category (str): Product category.
            
        Returns:
            dict: Product data.
        """
        try:
            # Delay to avoid being blocked
            time.sleep(random.uniform(0.5, 2))
            
            # Get product page
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract product name
            product_name_elem = soup.select_one('h1.listing-page-title-component')
            if not product_name_elem:
                logger.warning(f"No product title found for {url}")
                return None
                
            product_name = product_name_elem.text.strip()
            
            # Extract product description
            description = ""
            description_elem = soup.select_one('.listing-page-description')
            if description_elem:
                description = description_elem.text.strip()
            
            if not description:
                logger.warning(f"No description found for {product_name}")
                return None
            
            # Extract features (from item details)
            features = []
            feature_elems = soup.select('.listing-page-attributes .wt-display-block')
            for elem in feature_elems:
                features.append(elem.text.strip())
            
            # No explicit keywords on Etsy, so extract from tags
            keywords = ""
            tag_elems = soup.select('.tags a')
            if tag_elems:
                keywords = ', '.join([tag.text.strip() for tag in tag_elems])
            
            # Create product data dictionary
            product_data = {
                'product_name': product_name,
                'category': category,
                'description': description,
                'features': ' | '.join(features),
                'keywords': keywords,
                'url': url
            }
            
            logger.info(f"Scraped product: {product_name}")
            return product_data
            
        except Exception as e:
            logger.error(f"Error scraping product {url}: {e}")
            return None
    
    def save_to_csv(self, filename=None):
        """Save scraped products to CSV file.
        
        Args:
            filename (str): Name of the output file.
        """
        if not self.products:
            logger.warning("No products to save")
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.products)
        
        # Generate filename if not provided
        if not filename:
            timestamp = int(time.time())
            filename = os.path.join(self.output_dir, f"products_{timestamp}.csv")
        elif not os.path.isabs(filename):
            filename = os.path.join(self.output_dir, filename)
        
        # Save to CSV
        df.to_csv(filename, index=False)
        logger.info(f"Saved {len(self.products)} products to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Scrape product data from e-commerce sites')
    parser.add_argument('--source', type=str, choices=['amazon', 'etsy', 'both'], default='both',
                        help='E-commerce site to scrape (amazon, etsy, or both)')
    parser.add_argument('--categories', type=str, nargs='+', default=[
                        'laptop', 'smartphone', 'headphones', 'camera', 'watch', 
                        'backpack', 'sneakers', 'coffee maker', 'blender', 'air purifier'],
                        help='Categories to scrape')
    parser.add_argument('--max_products', type=int, default=50,
                        help='Maximum products to scrape per category')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file name')
    
    args = parser.parse_args()
    
    # Initialize scraper
    scraper = ProductScraper()
    
    # Scrape products
    if args.source in ['amazon', 'both']:
        scraper.scrape_amazon(args.categories, args.max_products)
    
    if args.source in ['etsy', 'both']:
        scraper.scrape_etsy(args.categories, args.max_products)
    
    # Save final results
    scraper.save_to_csv(args.output)
    
if __name__ == "__main__":
    main() 