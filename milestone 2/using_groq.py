import requests
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
from groq import Groq
from transformers import pipeline, AutoTokenizer
from eventregistry import *

# API Configuration
GROQ_API_KEY = "groq_api"
EVENT_REGISTRY_API_KEY = "Your_api"

# Model names
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

# Initialize Groq client
def initialize_groq():
    return Groq(api_key=GROQ_API_KEY)

def initialize_sentiment_analyzer():
    tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model=SENTIMENT_MODEL,
        tokenizer=tokenizer
    )
    return sentiment_pipeline, tokenizer

def truncate_for_model(text, tokenizer, max_length=512):
    """Truncate text to fit within model's token limit"""
    tokens = tokenizer.encode(text, truncation=False)
    if len(tokens) > max_length:
        tokens = tokens[:max_length-1] + [tokenizer.sep_token_id]
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text

def truncate_for_llama(text, max_length=900):
    """Truncate text for LLaMA model"""
    words = text.split()
    if len(words) > max_length:
        return ' '.join(words[:max_length]) + "..."
    return text

# Function to fetch news data from Event Registry
def fetch_news(max_items=100):
    try:
        # Initialize EventRegistry
        er = EventRegistry(apiKey=EVENT_REGISTRY_API_KEY)

        # Create query for articles
        q = QueryArticlesIter(
            keywords=QueryItems.OR([
                "Medicine", "Vaacines", "Counter drugs",
                "Drugs", "Medical Devices"
            ]),
            dateStart=(dt.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            dateEnd=dt.now().strftime('%Y-%m-%d'),
            dataType=["news", "blog"],
            lang="eng"
        )

        # Fetch articles with a limit on the number of results
        articles = []
        for article in q.execQuery(er, sortBy="date", maxItems=max_items):
            articles.append(article)

        return {"articles": {"results": articles}}
    except Exception as e:
        print(f"Error fetching news: {e}")
        return None

# Risk analysis with Groq LLaMa
def analyze_risk_with_llama(content, client):
    try:
        truncated_content = truncate_for_llama(content)

        prompt = f"""Analyze the following news article for Medicine supply chain risks.

        Consider these specific factors:
        1. Raw Material Risks
            - Availability of active pharmaceutical ingredients (APIs) and raw materials.
            - Price fluctuations in key ingredients (e.g., APIs, excipients, and packaging materials).
            - Dependency on a limited number of suppliers for essential raw materials.
            - Contamination risks in raw materials leading to safety and quality issues.

        2. Manufacturing Risks
            - Production capacity limitations during periods of high demand 
            - Delays due to technical issues in manufacturing lines
            - Disruptions in production caused by shortages in ancillary supplies
            - Dependency on contract manufacturers that may have competing priorities
            
        3. Geographic Risks
            - Heavy reliance on manufacturing hubs concentrated in specific regions 
            - Political instability or natural disasters disrupting supply in key regions
            - Logistical challenges in transporting temperature-sensitive products like vaccines
            - Regional supply-demand imbalances leading to product shortages

        4. Industry Impact
            - Shortages affecting critical healthcare systems and patient outcomes
            - Increased costs for healthcare providers and governments due to supply chain inefficiencies
            - Negative impacts on pharmaceutical companies reputations due to supply chain failures
            - Competitive disadvantages if rivals secure more robust supply chains

        5. Mitigation Strategies
            - Investment in alternative raw material sources and synthetic substitutes
            - Increasing in-house manufacturing capabilities to reduce dependence on third parties
            - Building strategic stockpiles of essential drugs, vaccines, and medical devices
            - Implementing advanced quality monitoring and predictive analytics in supply chains

        Article: {truncated_content}

        Provide a structured analysis of the identified risks and their potential impact on the Medicine supply chain."""

        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )

        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error with Groq LLaMa: {e}")
        return "Error in risk analysis"

# Sentiment analysis with proper truncation
def analyze_sentiment_with_model(content, sentiment_pipeline, tokenizer):
    try:
        # Properly truncate content for the model
        truncated_content = truncate_for_model(content, tokenizer)

        # Get sentiment prediction
        result = sentiment_pipeline(truncated_content)[0]

        # Format the result
        return {
            "label": result["label"],
            "score": float(result["score"]),
            "analysis": f"Sentiment: {result['label']} (confidence: {result['score']:.2f})"
        }
    except Exception as e:
        print(f"Error with sentiment analysis: {e}")
        return {
            "label": "ERROR",
            "score": 0.0,
            "analysis": "Error in sentiment analysis"
        }

# Aggregate data into structured format
def aggregate_data(news_data):
    try:
        structured_data = []
        for article in news_data.get('articles', {}).get('results', []):
            structured_data.append({
                "source": article.get('source', {}).get('title', ''),
                "title": article.get('title', ''),
                "description": article.get('body', ''),
                "content": article.get('body', ''),
                "published_at": article.get('dateTime', '')
            })
        return pd.DataFrame(structured_data)
    except Exception as e:
        print(f"Error structuring data: {e}")
        return None

# Main pipeline
def main():
    # Initialize models
    groq_client = initialize_groq()
    sentiment_pipeline, tokenizer = initialize_sentiment_analyzer()

    # Fetch news data
    news_data = fetch_news(max_items=10)
    if not news_data:
        return

    # Aggregate data into structured format
    structured_data = aggregate_data(news_data)
    if structured_data is None or structured_data.empty:
        print("No data to analyze")
        return

    # Analyze risk and sentiment
    results = []
    for idx, row in structured_data.iterrows():
        print(f"\nAnalyzing article {idx + 1}/{len(structured_data)}: {row['title']}")

        # Perform analyses
        risk_analysis = analyze_risk_with_llama(row['content'], groq_client)
        sentiment_analysis = analyze_sentiment_with_model(row['content'], sentiment_pipeline, tokenizer)

         # Store results
        results.append({
            'Title': row['title'],
            'Source': row['source'],
            'Published At': row['published_at'],
            'Sentiment': sentiment_analysis['label'],
            'Sentiment Score': sentiment_analysis['score'],
            'Sentiment Analysis': sentiment_analysis['analysis'],
            'Risk Analysis': risk_analysis
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("Risk_and_Sentiment_Results.csv", index=False, encoding='utf-8')

    print("Analysis saved to Risk_and_Sentiment_Results.csv")

if __name__ == "__main__":
    main()
