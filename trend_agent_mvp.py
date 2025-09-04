# Trend Identification Agent MVP
import re
from collections import Counter

# Basic list of English stop words for MVP
# A more comprehensive list would be used in a production system (e.g., from NLTK)
STOP_WORDS = set([
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", 
    "have", "has", "had", "do", "does", "did", "will", "would", "should", "can", 
    "could", "may", "might", "must", "am", "i", "you", "he", "she", "it", "we", 
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", 
    "their", "mine", "yours", "hers", "ours", "theirs", "to", "of", "in", "on", 
    "at", "by", "for", "with", "about", "against", "between", "into", "through", 
    "during", "before", "after", "above", "below", "from", "up", "down", "out", 
    "off", "over", "under", "again", "further", "then", "once", "here", "there", 
    "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
    "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", 
    "so", "than", "too", "very", "s", "t", "just", "don", "shouldn", "now", "ve", "ll", "d", "re"
])

def analyze_text_for_trends(text_data, top_n=5):
    """Analyzes text data for frequent keywords as a basic trend identification."""
    if not text_data:
        print("No text data provided for trend analysis.")
        return []

    print(f"Analyzing text for trends...")
    # Combine all text into a single string if it's a list
    if isinstance(text_data, list):
        text_data = " ".join(text_data)

    # 1. Basic Cleaning: Lowercase and remove punctuation
    text = text_data.lower()
    text = re.sub(r"[^a-z\s]", "", text) # Keep only letters and spaces

    # 2. Tokenization: Split into words
    words = text.split()

    # 3. Stop Word Removal
    filtered_words = [word for word in words if word not in STOP_WORDS and len(word) > 2] # Also filter short words

    # 4. Frequency Calculation
    word_counts = Counter(filtered_words)

    # 5. Get Top N frequent words
    most_common_words = word_counts.most_common(top_n)
    
    trends = []
    for word, count in most_common_words:
        trends.append({"keyword": word, "frequency": count})
        
    print(f"- Identified top {len(trends)} potential trend keywords.")
    return trends

if __name__ == "__main__":
    print("--- Trend Identification Agent MVP Running ---")
    
    # Sample data for MVP - in a real system, this would come from news summaries, social media, etc.
    sample_texts = [
        "The future of AI in healthcare is rapidly evolving with new machine learning models.",
        "Machine learning and AI are transforming the automotive industry, especially with autonomous driving.",
        "Experts discuss the impact of AI on job markets and the need for new skills.",
        "Investment in AI research continues to grow, focusing on ethical AI development.",
        "New AI algorithms are improving diagnostic accuracy in medical imaging.",
        "The rise of generative AI tools is a major topic in tech circles, with discussions on creative AI."
    ]
    
    identified_trends = analyze_text_for_trends(sample_texts, top_n=5)
    
    if identified_trends:
        print("\n--- Identified Potential Trends (MVP) --- ")
        for trend in identified_trends:
            print(f"Keyword: {trend['keyword'].ljust(15)} Frequency: {trend['frequency']}")
    else:
        print("No trends were identified from the sample data.")
        
    print("\n--- Trend Identification Agent MVP Finished ---")

