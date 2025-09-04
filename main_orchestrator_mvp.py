# Main Orchestrator MVP
import sys
import os
import json

# Adjust sys.path to allow imports from the parent directory (market_intelligence_ecosystem)
# This allows access to the agents module
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
GRANDPARENT_DIR = os.path.dirname(PARENT_DIR)
# Add the directory containing the 'agents' module to sys.path
# The structure is /home/ubuntu/market_intelligence_ecosystem, and this script is in orchestration subdir
# So, we need to add /home/ubuntu/market_intelligence_ecosystem to path
PROJECT_ROOT = PARENT_DIR # This should be /home/ubuntu/market_intelligence_ecosystem
sys.path.append(PROJECT_ROOT)

# Import agent functions
# It seems the original plan was to put each agent in its own sub-sub-directory.
# Let's adjust the path to reflect the actual structure used during development.
# The structure is: market_intelligence_ecosystem/agents/<agent_name_folder>/<agent_script.py>
# So, if PROJECT_ROOT is market_intelligence_ecosystem, then we need to import from agents.<agent_name_folder>.<agent_script_name_without_py>

try:
    from agents.news_summarization_agent.news_agent_mvp import fetch_and_summarize_news
    from agents.competitor_analysis_agent.competitor_agent_mvp import fetch_competitor_info
    from agents.trend_identification_agent.trend_agent_mvp import analyze_text_for_trends
    from agents.consumer_sentiment_analysis_agent.sentiment_agent_mvp import analyze_sentiment_basic
except ImportError as e:
    print(f"Error importing agent modules: {e}")
    print(f"Current sys.path: {sys.path}")
    print(f"PROJECT_ROOT used for sys.path.append: {PROJECT_ROOT}")
    # Attempting a more direct relative path for sys.path addition
    # This script is in /home/ubuntu/market_intelligence_ecosystem/orchestration
    # Agents are in /home/ubuntu/market_intelligence_ecosystem/agents
    # So, adding /home/ubuntu/market_intelligence_ecosystem should be correct.
    # Let's try to print the expected path for agent modules to debug
    expected_news_agent_path = os.path.join(PROJECT_ROOT, "agents", "news_summarization_agent", "news_agent_mvp.py")
    print(f"Expected path for news_agent_mvp.py: {expected_news_agent_path}")
    print(f"Does it exist? {os.path.exists(expected_news_agent_path)}")
    exit()

def run_ecosystem_mvp():
    """Runs a simple MVP workflow for the agent ecosystem."""
    print("--- Market Intelligence Ecosystem MVP Orchestrator Running ---")

    # 1. News Summarization Agent
    print("\nStep 1: Running News Summarization Agent...")
    # Using a default, known-working RSS feed for reliability in MVP
    news_items = fetch_and_summarize_news(rss_url="http://feeds.bbci.co.uk/news/rss.xml?edition=int", num_articles=2)
    if not news_items:
        print("News Summarization Agent failed to produce output. Exiting.")
        return

    print("\n--- News Summaries Received by Orchestrator ---")
    for item in news_items:
        print(json.dumps(item, indent=2))

    # Process the first news summary with other agents
    if news_items:
        first_summary_text = news_items[0].get("summary", "")
        if not first_summary_text or "No content available" in first_summary_text:
             # If summary is bad, try title
            first_summary_text = news_items[0].get("title", "")

        print(f"\nProcessing text for further analysis: ")
        print(f"\"...{first_summary_text[:100]}...\"")

        # 2. Trend Identification Agent (on the first news summary)
        print("\nStep 2: Running Trend Identification Agent on the first news summary...")
        trends = analyze_text_for_trends(first_summary_text, top_n=3)
        print("\n--- Trends Identified by Orchestrator ---")
        for trend in trends:
            print(json.dumps(trend, indent=2))

        # 3. Consumer Sentiment Analysis Agent (on the first news summary)
        print("\nStep 3: Running Consumer Sentiment Analysis Agent on the first news summary...")
        sentiment, score = analyze_sentiment_basic(first_summary_text)
        sentiment_result = {"text_analyzed_snippet": first_summary_text[:50]+"...", "sentiment": sentiment, "score_difference": score}
        print("\n--- Sentiment Analysis by Orchestrator ---")
        print(json.dumps(sentiment_result, indent=2))

    # 4. Competitor Analysis Agent (runs independently with a predefined URL for MVP)
    print("\nStep 4: Running Competitor Analysis Agent (on a predefined URL)...")
    # Using a placeholder that is generally accessible and was used in its MVP test
    competitor_url = "https://www.google.com/about/"
    competitor_info = fetch_competitor_info(competitor_url)
    if competitor_info:
        print("\n--- Competitor Info Received by Orchestrator ---")
        print(json.dumps(competitor_info, indent=2))
    else:
        print("Competitor Analysis Agent failed to produce output.")

    print("\n--- Market Intelligence Ecosystem MVP Orchestrator Finished ---")

if __name__ == "__main__":
    run_ecosystem_mvp()

