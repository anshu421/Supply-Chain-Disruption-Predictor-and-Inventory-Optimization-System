from flask import Flask, render_template, send_from_directory, request, redirect, url_for
from database import insert_product, fetch_all_products
from predict_disruption import run_prediction  # Import the function
from models import Product
from database import create_table
import pandas as pd
from predict_disruption import run_prediction  # Import the function

create_table()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/products')
def view_products():
    products = fetch_all_products()
    return render_template('products.html', products=products)

@app.route('/add_product', methods=['POST'])
def add_product():
    if request.method == 'POST':
        data = (
            request.form['Drug_Name'],
            request.form['Region'],
            request.form['Supplier_ID'],
            int(request.form['Demand']),
            int(request.form['Supply']),
            int(request.form['Lead_Time']),
            float(request.form['Transportation_Cost']),
            int(request.form['Product_Stock']),
            float(request.form['Cost_Price']),
            float(request.form['Selling_Price']),
            request.form['News_Article']
        )
        insert_product(data)
        return redirect(url_for('view_products'))
    return redirect(url_for('index'))

@app.route('/predict_disruptions', methods=['GET'])
def predict_disruptions():
    run_prediction()  # Running the prediction

    # Load the predicted results from CSV to display on the page
    results = pd.read_csv('inventory_predictions_with_utilization.csv')
    
    # Render the results in a table format on the 'results.html' page
    return render_template('results.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
