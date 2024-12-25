Supply Chain Disruption Predictor and Inventory Optimization System
Overview
This project is an AI-powered tool designed to monitor global supply chain data, predict disruptions, and optimize inventory levels. It performs sentiment analysis on supply chain news and provides insights to assist in proactive risk management.

Repository Structure
bash
Copy code
Supply-Chain-Disruption-Predictor-and-Inventory-Optimization-System/
├── Global_supply_chain.py                     # Main script for supply chain analysis
├── Medicine_supply_chain.py                   # Script focused on medicine-specific supply chain analysis
├── requirements.txt                           # Dependencies for the project
├── .gitignore                                 # Files and folders ignored by Git
├── data/
│   ├── medicines_supply_chain_news.csv        # Raw data for medicine supply chain
│   ├── medicines_supply_chain_news_with_sentiment.csv  # Sentiment-annotated data for medicines
│   ├── supply_chain_news.csv                  # Raw data for global supply chain
│   ├── supply_chain_news_with_sentiment.csv   # Sentiment-annotated global supply chain data
Features
Sentiment Analysis:

Performs sentiment analysis on supply chain news using analyze_sentiment function.
Visualizes sentiment distribution for better insights.
Data Processing:

Reads and processes CSV files for global and medicine-related supply chain data.
Saves sentiment-annotated data for further use.
Visualization:

Generates bar plots to display the distribution of sentiment in supply chain articles.
Setup and Installation
Prerequisites
Python 3.8 or higher
Virtual environment (optional but recommended)
Steps
Clone the repository:

bash
Copy code
git clone https://github.com/anshu421/Supply-Chain-Disruption-Predictor-and-Inventory-Optimization-System.git
cd Supply-Chain-Disruption-Predictor-and-Inventory-Optimization-System
Create and activate a virtual environment:

bash
Copy code
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the scripts:

For global supply chain analysis:
bash
Copy code
python Global_supply_chain.py
For medicine supply chain analysis:
bash
Copy code
python Medicine_supply_chain.py
Usage
Input Data: Place raw CSV files (e.g., supply_chain_news.csv, medicines_supply_chain_news.csv) in the appropriate directory before running the scripts.
Output Data:
Sentiment-annotated CSV files are generated automatically in the same directory.
Visualizations of sentiment distribution are displayed during execution.
Dependencies
All dependencies are listed in the requirements.txt file. Key libraries include:
pandas for data manipulation
matplotlib for data visualization
nltk or any sentiment analysis library for NLP
