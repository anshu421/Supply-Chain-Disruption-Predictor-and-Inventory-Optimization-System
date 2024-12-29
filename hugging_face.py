import requests
import pandas as pd
from textblob import TextBlob  # For sentiment analysis
import matplotlib.pyplot as plt

# Constants
NEWS_API_KEY = "75eadfae4ea8427888c43b4e89653a2b"  # Replace with your NewsAPI key
HUGGING_FACE_API_KEY = "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"  # Replace with your Hugging Face API key
BASE_URL_NEWS = "https://newsapi.org/v2/everything"
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-hf"

# Headers for Hugging Face API
HEADERS_HF = {
    "Authorization": f"Bearer {HUGGING_FACE_API_KEY}"
}

# Function to fetch news
def fetch_news(query="supply chain", language="en", page_size=100):
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    response = requests.get(BASE_URL_NEWS, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"Error fetching news: {response.status_code}")
        return []

# Function for sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze risk using LLaMA (Hugging Face)
def analyze_risk_with_llama(content):
    payload = {"inputs": content}
    try:
        response = requests.post(HUGGING_FACE_API_URL, headers=HEADERS_HF, json=payload)
        if response.status_code == 200:
            return response.json()[0]['generated_text']
        else:
            print(f"Error analyzing risk: {response.status_code}")
            return "Error in risk analysis."
    except Exception as e:
        print(f"Exception during risk analysis: {e}")
        return "Exception in risk analysis."

if __name__ == "__main__":
    # Fetch news articles
    articles = fetch_news()
    if not articles:
        print("No articles found.")
        exit()

    print("News articles fetched successfully!")

    # Create DataFrame
    news_data = [
        {
            "Title": article["title"],
            "Description": article.get("description", ""),
            "Source": article["source"]["name"],
            "Content": article.get("content", ""),
            "URL": article["url"],
        }
        for article in articles
    ]
    news_df = pd.DataFrame(news_data)

    # Ensure no None values in the Description column
    news_df["Description"] = news_df["Description"].fillna("")

    # Perform sentiment analysis
    print("Performing sentiment analysis...")
    news_df["Sentiment"] = news_df["Description"].apply(analyze_sentiment)

    # Perform risk analysis using LLaMA
    print("Performing risk analysis with LLaMA...")
    news_df["Risk Analysis"] = news_df["Content"].apply(analyze_risk_with_llama)

    # Save results to CSV
    news_df.to_csv("supply_chain_news_with_analysis.csv", index=False)
    print("Results saved to 'supply_chain_news_with_analysis.csv'.")


    # Sentiment distribution
    print("\nSentiment Distribution:")
    print(news_df["Sentiment"].value_counts())

    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    news_df["Sentiment"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Sentiment Analysis of Supply Chain News")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Articles")
    plt.show()

    # Aggregating risk factor
    print("\nAggregating Risk Factor...")
    negative_sentiment_count = news_df["Sentiment"].value_counts().get("Negative", 0)
    total_articles = len(news_df)
    risk_factor = (negative_sentiment_count / total_articles) * 100
    print(f"Risk Factor based on Negative Sentiment: {risk_factor:.2f}%")
