import requests
import pandas as pd
from textblob import TextBlob  # For sentiment analysis
import subprocess
import matplotlib.pyplot as plt

# Constants
NEWS_API_KEY = "75eadfae4ea8427888c43b4e89653a2b"  # Replace with your NewsAPI key
BASE_URL_NEWS = "https://newsapi.org/v2/everything"

# Function to fetch news
def fetch_news(query="supply chain", language="en", page_size=10):
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
    if not text.strip():  # Handle empty strings
        return "Neutral"
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to analyze risk with Ollama
def analyze_risk_with_ollama(content):
    if not content.strip():  # Ensure content is not empty
        return "No content to analyze."

    # Reduce input size to 300 characters
    content_snippet = content[:300]
    command = ["ollama", "run", "llama2", f"Analyze the risk in this content: {content_snippet}"]

    try:
        # Run the Ollama command with an increased timeout
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=60)  # Timeout set to 60 seconds
        
        if process.returncode != 0:  # Handle non-zero exit codes
            error_message = stderr.decode("utf-8").strip()
            print(f"Error analyzing risk with Ollama: {error_message}")
            return f"Error: {error_message}"
        
        # Return the analysis result
        return stdout.decode("utf-8").strip()
    
    except subprocess.TimeoutExpired:
        # Handle timeout explicitly
        process.kill()
        print("Timeout occurred in risk analysis.")
        return "Timeout in risk analysis."
    
    except Exception as e:
        # Catch and log any other exceptions
        error_message = f"Exception in risk analysis: {e}"
        print(error_message)
        return error_message

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

    # Perform risk analysis using Ollama
    print("Performing risk analysis with Ollama...")
    news_df["Risk Analysis"] = news_df["Content"].apply(analyze_risk_with_ollama)

    # Save results to CSV
    news_df.to_csv("supply_chain_news_with_analysis_Using_Model.csv", index=False)
    print("Results saved to 'supply_chain_news_with_analysis_Using_Model.csv'.")

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
