import re
# Basic sentiment keyword lists for MVP
# In a real system, this would be a sophisticated model (e.g., LLM-based or a trained classifier)
POSITIVE_WORDS = set([
    "good", "great", "excellent", "amazing", "fantastic", "love", "like", "happy", 
    "pleased", "satisfied", "wonderful", "awesome", "best", "perfect", "superb", "joy"
])
NEGATIVE_WORDS = set([
    "bad", "terrible", "horrible", "awful", "hate", "dislike", "sad", "unhappy",
    "disappointed", "poor", "worst", "problem", "issue", "fail", "error", "broken"
])
def analyze_sentiment_basic(text):
    """Performs basic sentiment analysis based on keyword matching."""
    if not text:
        return "neutral", 0 # Default to neutral if no text
    text_lower = text.lower()
    words = re.findall(r"\b\w+\b", text_lower) # Extract words
    positive_score = 0
    negative_score = 0
    for word in words:
        if word in POSITIVE_WORDS:
            positive_score += 1
        elif word in NEGATIVE_WORDS:
            negative_score += 1
    
    # Basic sentiment determination
    if positive_score > negative_score:
        sentiment = "positive"
    elif negative_score > positive_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"
        
    # A raw score for potential intensity (very basic)
    score_difference = positive_score - negative_score
    
    return sentiment, score_difference
if __name__ == "__main__":
    print("--- Consumer Sentiment Analysis Agent MVP Running ---")
    
    # Sample social media-like text data for MVP
    sample_posts = [
        "I absolutely love the new features on this product! It's amazing.",
        "This update is terrible, my app keeps crashing. Very disappointed.",
        "The customer service was okay, neither good nor bad.",
        "What a fantastic experience! I am so happy with my purchase.",
        "I hate how slow this has become. It used to be much better.",
        "The product is fine, it does what it says.",
        "This is the best thing I have bought all year! Superb quality.",
        "I encountered a problem with the installation, it was quite frustrating."
    ]
    
    analyzed_sentiments = []
    print(f"Analyzing {len(sample_posts)} sample posts...")
    for i, post in enumerate(sample_posts):
        sentiment, score = analyze_sentiment_basic(post)
        analyzed_sentiments.append({
            "post_id": i + 1,
            "text": post,
            "sentiment": sentiment,
            "score_difference": score
        })
        print(f"- Post {i+1}: Sentiment - {sentiment}")
        
    if analyzed_sentiments:
        print("\n--- Analyzed Sentiments (MVP) --- ")
        for item in analyzed_sentiments:
            print(f"\nPost ID: {item['post_id']}")
            print(f"Text: {item['text']}")
            print(f"Sentiment: {item['sentiment']} (Score Diff: {item['score_difference']})")
    else:
        print("No sentiments were analyzed from the sample data.")
        
    print("\n--- Consumer Sentiment Analysis Agent MVP Finished ---")