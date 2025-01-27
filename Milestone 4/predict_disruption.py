from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Download necessary NLTK data
nltk.download('vader_lexicon')

def send_email(subject, body, recipient_email):
    sender_email = "footballdhoni@gmail.com"
    sender_password = "crro cqkm eaqb vucp"  # It's better to use app-specific passwords for security

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            text = msg.as_string()
            server.sendmail(sender_email, recipient_email, text)
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def fetch_data_from_db():
    # Connect to the SQLite database
    conn = sqlite3.connect('inventory.db')
    query = '''
        SELECT Product_ID, Drug_Name, Region, Supplier_ID, Demand, Supply, Lead_Time, Transportation_Cost, Product_Stock, Cost_Price, Selling_Price, News_Article
        FROM products
    '''
    data = pd.read_sql(query, conn)
    conn.close()
    return data

def run_prediction():
    # Load data from the database
    data = fetch_data_from_db()

    # Feature Engineering: Demand-Supply Gap
    data['Demand_Supply_Gap'] = data['Demand'] - data['Supply']

    # Risk Classification
    def classify_risk(row):
        if row['Demand_Supply_Gap'] > 100 or row['Lead_Time'] > 15:
            return 'High'
        elif row['Demand_Supply_Gap'] > 50 or row['Lead_Time'] > 10:
            return 'Medium'
        else:
            return 'Low'
        
    data['Risk_Level'] = data.apply(classify_risk, axis=1)

    
    # Encode Categorical Features
    risk_label_encoder = LabelEncoder()
    data['Risk_Level_Encoded'] = risk_label_encoder.fit_transform(data['Risk_Level'])
    data['Region_Encoded'] = LabelEncoder().fit_transform(data['Region'])
    data['Drug_Encoded'] = LabelEncoder().fit_transform(data['Drug_Name'])

    # Risk Prediction Model
    X_risk = data[['Demand', 'Supply', 'Lead_Time', 'Transportation_Cost', 'Region_Encoded', 'Drug_Encoded']]
    y_risk = data['Risk_Level_Encoded']
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(X_risk, y_risk, test_size=0.2, random_state=42)
    risk_model = XGBClassifier(eval_metric='mlogloss')
    risk_model.fit(X_train_risk, y_train_risk)
    data['Predicted_Risk_Level'] = risk_label_encoder.inverse_transform(risk_model.predict(X_risk))

    # Sentiment Analysis on News Articles
    data['News_Article'] = data['News_Article'].fillna("No content available")
    data['News_Article'] = data['News_Article'].apply(lambda x: re.sub(r'\W+', ' ', str(x)).lower())

    def analyze_sentiment_vader(text):
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(str(text))
        # Scale the compound score to the range [0, 1] and round to 5 decimal places
        scaled_score = round((score['compound'] + 1) / 2, 5)
        return scaled_score

    data['Sentiment'] = data['News_Article'].apply(analyze_sentiment_vader)

    # Add Warehouse Capacity
    data['Warehouse_Capacity'] = 1000  # Example: Warehouse capacity for all rows

    # Inventory Prediction Logic with Utilization
    def predict_inventory_and_utilization(row):
        if row['Predicted_Risk_Level'] == 'High':
            predicted_inventory = row['Demand'] + (0.8 * row['Warehouse_Capacity'])
        elif row['Predicted_Risk_Level'] == 'Medium':
            predicted_inventory = row['Demand'] + (0.5 * row['Warehouse_Capacity'])
        elif row['Predicted_Risk_Level'] == 'Low':
            predicted_inventory = row['Demand'] - (0.2 * row['Warehouse_Capacity'])
        else:
            predicted_inventory = row['Demand']
        
        utilization_percentage = (predicted_inventory / row['Warehouse_Capacity']) * 100
        return pd.Series([predicted_inventory, utilization_percentage])

    data[['Predicted_Inventory', 'Utilization_Percentage']] = data.apply(predict_inventory_and_utilization, axis=1)

    # Add a timestamp for the last update
    data['Last_Updated'] = datetime.now().strftime('%Y-%m-%d')

    # Visualizations
    def create_demand_supply_gap_plot():
        plt.figure(figsize=(8, 6))
        sns.histplot(data['Demand_Supply_Gap'], kde=True, bins=20)
        plt.title('Demand-Supply Gap Distribution')
        plt.xlabel('Demand-Supply Gap')
        plt.ylabel('Frequency')
        
        # Save plot to a BytesIO object and return as base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        demand_supply_gap_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return demand_supply_gap_plot

    def create_risk_level_plot():
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Risk_Level', data=data)
        plt.title('Risk Level Distribution')
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        
        # Save plot to a BytesIO object and return as base64
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        risk_level_plot = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return risk_level_plot

    # Generate the plots
    demand_supply_gap_plot = create_demand_supply_gap_plot()
    risk_level_plot = create_risk_level_plot()

    # Save the updated dataset with predictions to CSV
    data.to_csv('inventory_predictions_with_utilization.csv', index=False)

    print("Inventory predictions with utilization have been updated.")

    # Check if any critical risks (High risk level)
    critical_risk_rows = data[data['Risk_Level'] == 'High']

    if not critical_risk_rows.empty:
     subject = "ALERT: Critical Supply Chain Disruption Detected"
    
    body = "ALERT: The following high-risk supply chain disruptions have been detected:\n\n"
    
    for index, row in critical_risk_rows.iterrows():
        body += f"Product ID: {row['Product_ID']}\n"
        body += f"Drug Name: {row['Drug_Name']}\n"
        body += f"Region: {row['Region']}\n"
        body += f"Supplier ID: {row['Supplier_ID']}\n"
        body += f"Demand: {row['Demand']}\n"
        body += f"Supply: {row['Supply']}\n"
        body += f"Predicted Risk Level: {row['Predicted_Risk_Level']}\n"
        body += f"Utilization Percentage: {row['Utilization_Percentage']:.2f}%\n"
        body += f"Predicted Inventory: {row['Predicted_Inventory']}\n\n"
        body += "-"*50 + "\n"  # Separator between product details
    
    recipient_email = "team_email@example.com"  # Replace with actual recipient email
    send_email(subject, body, recipient_email)


    # Send an email with the relevant information after the predictions are made
    subject = "Updated Inventory Predictions"
    body = "Here are the latest inventory predictions:\n\n"
    
    for index, row in data.iterrows():
        body += f"Product ID: {row['Product_ID']}\n"
        body += f"Drug Name: {row['Drug_Name']}\n"
        body += f"Region: {row['Region']}\n"
        body += f"Supplier ID: {row['Supplier_ID']}\n"
        body += f"Demand: {row['Demand']}\n"
        body += f"Supply: {row['Supply']}\n"
        body += f"Predicted Risk Level: {row['Predicted_Risk_Level']}\n"
        body += f"Utilization Percentage: {row['Utilization_Percentage']:.2f}%\n"
        body += f"Predicted Inventory: {row['Predicted_Inventory']}\n\n"
    
    recipient_email = "team_email@example.com"  # Replace with the actual recipient's email address
    send_email(subject, body, recipient_email)

    return demand_supply_gap_plot, risk_level_plot
