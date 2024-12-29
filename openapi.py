import requests
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline
import openai
import os

# Ensure your API keys are set as environment variables or hardcoded (not recommended for production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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")  # Replace with your actual key
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "75eadfae4ea8427888c43b4e89653a2b")  # Replace with your actual key

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Hugging Face LLaMA pipeline initialization
def initialize_llama_pipeline():
    try:
        return pipeline("text-classification", model="facebook/bart-large-mnli")
    except Exception as e:
        print(f"Error initializing Hugging Face pipeline: {e}")
        return None

# Function to fetch news articles
def fetch_news(query="supply chain", language="en", page_size=100):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        return articles
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []

# Function to perform sentiment analysis with TextBlob
def analyze_sentiment_with_textblob(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to perform risk analysis with OpenAI GPT
def analyze_risk_with_gpt(content):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in supply chain risk analysis."},
                {"role": "user", "content": content},
            ],
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error analyzing risk with GPT: {e}")
        return "No analysis available."

# Main script
if __name__ == "__main__":
    # Initialize Hugging Face pipeline
    llama_pipeline = initialize_llama_pipeline()
    if llama_pipeline is None:
        print("Skipping Hugging Face sentiment analysis due to initialization failure.")

    # Fetch news articles
    print("Fetching news articles...")
    articles = fetch_news()
    if not articles:
        print("No articles found.")
    else:
        print(f"Fetched {len(articles)} articles.")

        # Process articles and perform analyses
        news_data = []
        for article in articles:
            title = article.get("title", "No Title")
            description = article.get("description", "No Description")
            source = article.get("source", {}).get("name", "Unknown")
            url = article.get("url", "No URL")
            sentiment = analyze_sentiment_with_textblob(description)

            # Perform risk analysis using GPT
            print(f"\nAnalyzing article: {title}")
            risk_analysis = analyze_risk_with_gpt(description)

            # Sentiment analysis with LLaMA
            llama_sentiment = "No analysis" if llama_pipeline is None else llama_pipeline(description)

            news_data.append({
                "Title": title,
                "Description": description,
                "Source": source,
                "URL": url,
                "Sentiment (TextBlob)": sentiment,
                "Risk Analysis (GPT)": risk_analysis,
                "Sentiment (LLaMA)": llama_sentiment,
            })

        # Create a DataFrame
        news_df = pd.DataFrame(news_data)

        # Save results to CSV
        news_df.to_csv("supply_chain_news_analysis.csv", index=False)
        print("\nResults saved to supply_chain_news_analysis.csv")

        # Display sentiment distribution
        print("\nSentiment Distribution (TextBlob):")
        print(news_df["Sentiment (TextBlob)"].value_counts())

        # Plot sentiment distribution
        plt.figure(figsize=(8, 6))
        news_df["Sentiment (TextBlob)"].value_counts().plot(kind="bar", color="skyblue")
        plt.title("Sentiment Analysis of Supply Chain News (TextBlob)")
        plt.xlabel("Sentiment")
        plt.ylabel("Number of Articles")
        plt.show()

        # Aggregate Risk Factor
        print("\nAggregating Risk Factor...")
        risk_factor = news_df["Sentiment (TextBlob)"].value_counts(normalize=True).get("Negative", 0) * 100
        print(f"Calculated Risk Factor from Negative Sentiment: {risk_factor:.2f}%")
