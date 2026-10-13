# News Summarization Agent MVP
import feedparser
import re

# Placeholder for a more sophisticated LLM-based summarization function
def summarize_text_basic(text, num_sentences=2):
    """A very basic summarizer: returns the first num_sentences sentences."""
    if not text:
        return "No content available for summarization."
    # A simple way to split into sentences, might not be perfect for all cases
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    summary = " ".join(sentences[:num_sentences])
    if len(sentences) > num_sentences:
        summary += "..."
    return summary

def fetch_and_summarize_news(rss_url="http://news.google.com/news?ned=us&topic=h&output=rss", num_articles=3):
    """Fetches news from an RSS feed and provides basic summaries."""
    print(f"Fetching news from: {rss_url}")
    feed = feedparser.parse(rss_url)

    if feed.bozo:
        print(f"Error parsing RSS feed: {feed.bozo_exception}")
        return []

    summarized_news = []
    print(f"Found {len(feed.entries)} articles. Processing up to {num_articles}...")

    for i, entry in enumerate(feed.entries):
        if i >= num_articles:
            break
        
        title = entry.title if hasattr(entry, 'title') else "No Title"
        link = entry.link if hasattr(entry, 'link') else "No Link"
        
        # Try to get content for summarization from 'summary' or 'description'
        content_to_summarize = ""
        if hasattr(entry, 'summary'):
            content_to_summarize = entry.summary
        elif hasattr(entry, 'description'):
            content_to_summarize = entry.description

        # Basic cleaning: remove HTML tags for summarization
        cleaned_content = re.sub('<[^<]+?>', '', content_to_summarize) # Basic HTML tag removal
        
        # In a real scenario, this would be a call to an LLM API
        # summary = llm_service.summarize(cleaned_content)
        summary = summarize_text_basic(cleaned_content)
        
        summarized_news.append({
            "title": title,
            "link": link,
            "summary": summary
        })
        print(f"- Processed: {title}")
        
    return summarized_news

if __name__ == "__main__":
    print("--- News Summarization Agent MVP Running ---")
    # Example usage:
    # Using a general Google News RSS feed for demonstration
    # In a real application, this URL would be configurable and potentially multiple feeds would be used.
    news_items = fetch_and_summarize_news()
    
    if news_items:
        print("\n--- Summarized News --- ")
        for item in news_items:
            print(f"\nTitle: {item['title']}")
            print(f"Link: {item['link']}")
            print(f"Summary (basic): {item['summary']}")
    else:
        print("No news items were processed.")
    print("\n--- News Summarization Agent MVP Finished ---")

