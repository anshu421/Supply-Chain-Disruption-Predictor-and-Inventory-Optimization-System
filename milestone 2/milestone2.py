# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Load the dataset
data = pd.read_csv('pharmaceutical_dataset_2.csv')

# Feature Engineering: Demand-Supply Gap
data['Demand_Supply_Gap'] = data['Demand'] - data['Supply']

# Risk Classification Function
def classify_risk(row):
    if row['Demand_Supply_Gap'] > 100 or row['Lead_Time'] > 15:
        return 'High'
    elif row['Demand_Supply_Gap'] > 50 or row['Lead_Time'] > 10:
        return 'Medium'
    else:
        return 'Low'

# Apply Risk Classification
data['Risk_Level'] = data.apply(classify_risk, axis=1)

# Encode Categorical Features
risk_label_encoder = LabelEncoder()
data['Risk_Level_Encoded'] = risk_label_encoder.fit_transform(data['Risk_Level'])
data['Region_Encoded'] = LabelEncoder().fit_transform(data['Region'])
data['Drug_Encoded'] = LabelEncoder().fit_transform(data['Drug_Name'])

# ---------------------------- XGBOOST RISK CLASSIFICATION MODEL ---------------------------- #

# Define Features and Target
X_risk = data[['Demand', 'Supply', 'Lead_Time', 'Transportation_Cost', 'Region_Encoded', 'Drug_Encoded']]
y_risk = data['Risk_Level_Encoded']

# Split Data into Training and Testing Sets
X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)

# Initialize & Train XGBoost Model
risk_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, eval_metric='mlogloss')
risk_model.fit(X_train_risk, y_train_risk)

# Predict on Test Data
risk_predictions = risk_model.predict(X_test_risk)

# Evaluate Model Performance
print("Risk Classification Accuracy:", accuracy_score(y_test_risk, risk_predictions))

# Assign Predicted Risk Level to Dataset
data['Predicted_Risk_Level'] = risk_label_encoder.inverse_transform(risk_model.predict(X_risk))

# ---------------------------- SENTIMENT ANALYSIS ON NEWS ARTICLES ---------------------------- #

# Handle Missing News Articles
data['News_Article'] = data['News_Article'].fillna("No content available")
data['News_Article'] = data['News_Article'].apply(lambda x: re.sub(r'\W+', ' ', str(x)).lower())

# Function to Analyze Sentiment
def analyze_sentiment_vader(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(str(text))
    return round((score['compound'] + 1) / 2, 5)  # Normalize to range [0,1]

# Apply Sentiment Analysis
data['Sentiment'] = data['News_Article'].apply(analyze_sentiment_vader)

# ---------------------------- SAVE UPDATED DATASET ---------------------------- #
data.to_csv('updated_pharmaceutical_xgboost.csv', index=False)
print("Updated dataset saved successfully!")
