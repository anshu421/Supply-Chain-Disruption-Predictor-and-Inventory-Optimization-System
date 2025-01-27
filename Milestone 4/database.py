import sqlite3
import csv

def create_table():
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS products (
            Product_ID INTEGER PRIMARY KEY AUTOINCREMENT,
            Drug_Name TEXT NOT NULL,
            Region TEXT NOT NULL,
            Supplier_ID INTEGER NOT NULL,
            Demand INTEGER NOT NULL,
            Supply INTEGER NOT NULL,
            Lead_Time INTEGER NOT NULL,
            Transportation_Cost REAL NOT NULL,
            Product_Stock INTEGER NOT NULL,
            Cost_Price REAL NOT NULL,
            Selling_Price REAL NOT NULL,
            News_Article TEXT
        )
    ''')
    connection.commit()
    connection.close()

def insert_product(product):
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()
    try:
        cursor.execute('''
            INSERT INTO products (
                Drug_Name, Region, Supplier_ID, Demand, Supply, Lead_Time,
                Transportation_Cost, Product_Stock, Cost_Price, Selling_Price, News_Article
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', product)
        connection.commit()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        connection.close()

def fetch_all_products():
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM products')
    products = cursor.fetchall()
    connection.close()
    return products

def insert_products_from_csv(csv_filename):
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()

def insert_products_from_csv(csv_filename):
    connection = sqlite3.connect('inventory.db')
    cursor = connection.cursor()

    # Read the CSV and insert each row into the database
    with open(csv_filename, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)  # Assuming the CSV has headers
        for row in reader:
            product = (
                row['Drug_Name'],
                row['Region'],
                row['Supplier_ID'],  # Keep as string
                int(row['Demand']),
                int(row['Supply']),
                int(row['Lead_Time']),
                float(row['Transportation_Cost']),
                int(float(row['Product_Stock'])),  # Convert Product_Stock from float to int
                float(row['Cost_Price']),
                float(row['Selling_Price']),
                row['News_Article']
            )
            insert_product(product)

    connection.close()
    print(f"Products from {csv_filename} have been inserted into the database.")
