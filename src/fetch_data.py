import sqlite3
import pandas as pd

def fetch_selected_data(source_db_path, table_name, selected_columns):
    conn = sqlite3.connect(source_db_path)
    cursor = conn.cursor()
    columns_str = ','.join(selected_columns)
    query = f"SELECT {columns_str} FROM {table_name}"
    cursor.execute(query)
    data = pd.DataFrame(cursor.fetchall(), columns=selected_columns)
    conn.close()
    return data