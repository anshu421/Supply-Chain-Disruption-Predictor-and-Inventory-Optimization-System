import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import classification_report, mean_squared_error

# Load the dataset
data = pd.read_csv('pharmaceutical_dataset.csv')

# Feature engineering for risk factors
# Calculate demand-supply gap as a new feature
data['Demand_Supply_Gap'] = data['Demand'] - data['Supply']

# Define a risk label based on custom thresholds (example)
def classify_risk(row):
    if row['Demand_Supply_Gap'] > 100 or row['Lead_Time'] > 15:
        return 'High'
    elif row['Demand_Supply_Gap'] > 50 or row['Lead_Time'] > 10:
        return 'Medium'
    else:
        return 'Low'

data['Risk_Level'] = data.apply(classify_risk, axis=1)

# Encode categorical features
label_encoder = LabelEncoder()
data['Region_Encoded'] = label_encoder.fit_transform(data['Region'])
data['Drug_Encoded'] = label_encoder.fit_transform(data['Drug_Name'])

# Prepare features and labels for risk classification
X = data[['Demand', 'Supply', 'Lead_Time', 'Transportation_Cost', 'Region_Encoded', 'Drug_Encoded']]
y_risk = label_encoder.fit_transform(data['Risk_Level'])

# Split data
X_train, X_test, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)

# Train an XGBoost classifier for risk prediction
risk_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
risk_model.fit(X_train, y_train_risk)

# Predict risk levels for the entire dataset
data['Predicted_Risk_Level'] = risk_model.predict(X)  # Use integer values for risk

# Evaluate the model
risk_predictions = risk_model.predict(X_test)

# Save the classification report to a file
with open('risk_classification_report.txt', 'w') as report_file:
    report_file.write("Risk Classification Report:\n")
    report_file.write(classification_report(y_test_risk, risk_predictions, target_names=label_encoder.classes_))

# Save model if needed
# risk_model.save_model('risk_xgboost_model.json')

# Save the updated dataset with predicted risk levels
data.to_csv('pharmaceutical_dataset_with_risk.csv', index=False)
