import re
import string
import logging

logger = logging.getLogger(__name__)

def extract_keywords(text, min_length=3, max_keywords=10):
    """Extract potential keywords from text.
    
    Args:
        text (str): Input text to analyze.
        min_length (int): Minimum length of keywords.
        max_keywords (int): Maximum number of keywords to return.
        
    Returns:
        list: List of potential keywords.
    """
    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Split by whitespace
    words = text.split()
    
    # Count word frequency
    word_counts = {}
    for word in words:
        if len(word) >= min_length:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
    
    # Sort by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Get top keywords
    keywords = [word for word, count in sorted_words[:max_keywords]]
    
    return keywords

def analyze_seo(description, target_keywords):
    """Analyze a description for SEO effectiveness.
    
    Args:
        description (str): Description to analyze.
        target_keywords (list): List of target keywords.
        
    Returns:
        dict: SEO analysis results.
    """
    # Convert text to lowercase for case-insensitive matching
    lower_description = description.lower()
    
    # Count keyword occurrences
    keyword_counts = {}
    for keyword in target_keywords:
        # Count exact matches
        exact_matches = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', lower_description))
        keyword_counts[keyword] = exact_matches
    
    # Calculate keyword density
    total_words = len(description.split())
    keyword_density = sum(keyword_counts.values()) / total_words if total_words > 0 else 0
    
    # Check if description has good length (50-160 words is often recommended)
    length_score = 1 if 50 <= total_words <= 160 else 0.5
    
    # Check keyword presence
    keywords_found = sum(1 for count in keyword_counts.values() if count > 0)
    keyword_presence_score = keywords_found / len(target_keywords) if target_keywords else 0
    
    # Overall SEO score (simple weighted average)
    seo_score = (0.4 * keyword_presence_score + 0.3 * keyword_density + 0.3 * length_score) * 100
    
    return {
        'keyword_counts': keyword_counts,
        'keyword_density': keyword_density,
        'total_words': total_words,
        'keywords_found': keywords_found,
        'length_score': length_score,
        'keyword_presence_score': keyword_presence_score,
        'seo_score': seo_score
    }

def suggest_improvements(seo_analysis, target_keywords):
    """Suggest improvements for SEO.
    
    Args:
        seo_analysis (dict): Results from analyze_seo.
        target_keywords (list): List of target keywords.
        
    Returns:
        list: Suggested improvements.
    """
    suggestions = []
    
    # Check word count
    if seo_analysis['total_words'] < 50:
        suggestions.append("Description is too short. Aim for at least 50 words.")
    elif seo_analysis['total_words'] > 160:
        suggestions.append("Description might be too long. Consider shortening to 160 words or less.")
    
    # Check keyword density
    if seo_analysis['keyword_density'] < 0.01:
        suggestions.append("Keyword density is low. Try to naturally include more relevant keywords.")
    elif seo_analysis['keyword_density'] > 0.05:
        suggestions.append("Keyword density is high, which might look like keyword stuffing. Reduce keyword repetition.")
    
    # Check missing keywords
    missing_keywords = [keyword for keyword in target_keywords if seo_analysis['keyword_counts'].get(keyword, 0) == 0]
    if missing_keywords:
        suggestions.append(f"Missing keywords: {', '.join(missing_keywords)}. Try to naturally include them.")
    
    # Check overused keywords
    overused_keywords = [keyword for keyword, count in seo_analysis['keyword_counts'].items() if count > 3]
    if overused_keywords:
        suggestions.append(f"Overused keywords: {', '.join(overused_keywords)}. Reduce repetition.")
    
    return suggestions

def optimize_description(description, target_keywords):
    """Attempt to optimize a description for SEO.
    
    This is a simple implementation. In practice, this would be more sophisticated,
    potentially using NLP techniques to ensure natural language while incorporating keywords.
    
    Args:
        description (str): Description to optimize.
        target_keywords (list): List of target keywords.
        
    Returns:
        str: Optimized description.
    """
    # Analyze current SEO performance
    analysis = analyze_seo(description, target_keywords)
    
    # If SEO score is already good, return as is
    if analysis['seo_score'] > 80:
        return description
    
    # Find missing keywords
    missing_keywords = [keyword for keyword in target_keywords if analysis['keyword_counts'].get(keyword, 0) == 0]
    
    # Simple substitution - replace common words with keywords
    # This is a very basic approach; a real implementation would be more sophisticated
    optimized = description
    
    if missing_keywords:
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', optimized)
        
        # Add missing keywords to appropriate sentences
        for i, keyword in enumerate(missing_keywords):
            if i < len(sentences):
                # Simple replacement - this is naive but illustrates the concept
                words = sentences[i].split()
                if len(words) > 5:  # Only modify longer sentences
                    # Replace a common word with the keyword
                    for j, word in enumerate(words):
                        if len(word) > 3 and word.lower() not in target_keywords:
                            words[j] = keyword
                            break
                    sentences[i] = ' '.join(words)
        
        # Rejoin sentences
        optimized = ' '.join(sentences)
    
    return optimized 