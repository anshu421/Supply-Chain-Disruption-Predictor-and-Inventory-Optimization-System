Inventory Management and Risk Prediction Application
Overview
This application helps manage pharmaceutical inventory, predict risks based on supply-demand data, analyze sentiments from news articles related to drugs, and calculate inventory levels based on warehouse capacity. It provides a web interface for adding products, viewing inventory, and running predictions to identify potential risks and utilization.
________________________________________
Features
       •Add and view product inventory details through a web interface.
	•Predict supply chain disruptions and classify risk levels.
	•Perform sentiment analysis on news articles related to drugs.
	•Predict inventory levels based on warehouse capacity and risk classification.
	•Calculate warehouse utilization percentage.
	•Export data to CSV for further analysis.
________________________________________
File Descriptions
1. app.py
This is the main Flask application that serves the web interface. It includes:
	•Routes to display the homepage, product list, and prediction results.
	•Integration with the database and prediction logic.
	•Provides an endpoint to run the prediction and display results.

2. upload.py
Used to upload bulk product data into the database from a CSV file.
	•Parses the CSV file and stores the data in inventory.db.
	•Useful for initializing the database with large datasets.

3. predict_disruption.py
Handles the prediction logic:
	•Connects to the database to fetch inventory data.
	•Calculates demand-supply gap, risk levels, and performs sentiment analysis on news articles.
	•Predicts inventory levels based on warehouse capacity and risk classification:
o High Risk: Allocates 80% of warehouse capacity as additional inventory.
o Medium Risk: Allocates 50% of warehouse capacity as additional inventory.
o Low Risk: Reduces inventory by 20% of warehouse capacity.
	•Calculates warehouse utilization percentage.
	•Saves the results, including inventory predictions, risk levels, and utilization, to predicted_results.csv.

4. database.py
Manages the database operations:
	•Contains functions to create the database tables (inventory.db) if they don’t exist.
	•Allows inserting products and fetching product data.
	•Centralizes database-related tasks for easy management.

5. database.db
This is the database file used for user and authentication-related data (if applicable). Not directly used for inventory in this application.

6. inventory.db
The primary database file storing inventory-related data. It contains details such as:
	•Drug Name, Region, Supplier ID, Demand, Supply, Lead Time, and other essential information.
	•Serves as the data source for predictions and web interface operations.

7. export_to_csv.py
Provides functionality to export database data into a CSV file:
	•Useful for generating reports or backing up inventory data.
	•Can be run independently to create a snapshot of the database.
________________________________________
How to Run the Application
1. Setup
	•Ensure you have Python 3.7+ installed on your system.
	•Install the required dependencies:
bash
pip install -r requirements.txt
2. Database Initialization
	•Ensure inventory.db exists. If not, it will be created automatically when you run the application.
	•To upload bulk data, use upload.py:
bash
python upload.py
3. Run the Application
	•Start the Flask server:
bash
python app.py
	•Open your browser and go to http://127.0.0.1:5000 to access the application.

4. Using the Web Interface
	•Add Products: Use the form on the "Add Product" page to add inventory details.
	•View Products: Navigate to the "Products" page to see the inventory list.
	•Predict Disruptions: Click the "Run Prediction" button to analyze risks, predict inventory levels, and calculate warehouse utilization. The results will be saved to 							predicted_results.csv and displayed on the "Results" page.
5. Export Data
	•To export inventory data from inventory.db to a CSV file, run:
bash
python export_to_csv.py
________________________________________
Sample Workflow
1.Upload Data: Use upload.py to populate inventory.db.
2.Run App: Start the Flask app with app.py.
3.Add Products: Add or edit inventory data using the web interface.
4.Predict Risks and Inventory Levels:
	oAnalyze risks by clicking "Run Prediction."
	oPredictions will include:
	Risk Levels (High, Medium, Low).
	Predicted Inventory Levels based on warehouse capacity.
	Warehouse Utilization Percentage.
	oResults are displayed on the "Results" page and saved to predicted_results.csv.

5.Export Data: Generate a CSV report using export_to_csv.py.

