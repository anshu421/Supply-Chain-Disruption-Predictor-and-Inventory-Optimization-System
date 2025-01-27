import sqlite3
import csv

# Function to check if table exists in the database
def table_exists(cursor, table_name):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cursor.fetchone() is not None

# Function to create the 'products' table if it doesn't exist
def create_products_table(cursor):
    cursor.execute('''CREATE TABLE IF NOT EXISTS products (
                        Product_ID INTEGER PRIMARY KEY,
                        Drug_Name TEXT,
                        Region TEXT,
                        Supplier_ID TEXT,
                        Demand INTEGER,
                        Supply INTEGER,
                        Lead_Time INTEGER,
                        Transportation_Cost REAL,
                        Product_Stock INTEGER,
                        Cost_Price REAL,
                        Selling_Price REAL,
                        News_Article TEXT
                    )''')

# Function to export data from database to CSV
def export_to_csv(db_file, table_name, csv_filename):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if the table exists
    if not table_exists(cursor, table_name):
        print(f"Table '{table_name}' does not exist in '{db_file}' database.")
        conn.close()
        return
    
    # Fetch data from the specified table
    cursor.execute(f"SELECT * FROM {table_name}")
    
    # Get column names for the CSV header
    columns = [description[0] for description in cursor.description]
    
    # Write the data to CSV
    with open(csv_filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)  # Write the header
        writer.writerows(cursor.fetchall())  # Write the rows
    
    print(f"Data from '{table_name}' has been exported to '{csv_filename}'.")

    # Close the connection
    conn.close()

# Export from 'inventory.db' and 'database.db'
export_to_csv('inventory.db', 'products', 'products_inventory.csv')
export_to_csv('database.db', 'products', 'products_database.csv')
