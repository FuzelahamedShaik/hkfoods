from flask import Flask, render_template, jsonify, request, redirect
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

app = Flask(__name__)

# Load data from the Excel file into pandas DataFrames
file_path = '../HKFoods_Hackathon data 25102024.xlsx'
excel_data = pd.ExcelFile(file_path)

# Read the relevant sheets
hope_production = excel_data.parse('HOPE PRODUCTION')
hope_storage_after_cooking = excel_data.parse('HOPE STORAGE AFTER COOKING')
faith_production = excel_data.parse('FAITH PRODUCTION')
faith_storage_after_cooking = excel_data.parse('FAITH STORAGE AFTER COOKING')
package_weights = excel_data.parse('HOPE-FAITH PACKAGE WEIGHTS')

# Initialize database with thresholds table
def init_db():
    # Connecting to the SQLite database (create if it doesn't exist)
    conn = sqlite3.connect('dashboard.db')
    cursor = conn.cursor()

    # Create tables if they do not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS production_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        product TEXT,
                        weight_deviation REAL,
                        humidity REAL,
                        temperature REAL,
                        team TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS team_scores (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team TEXT,
                        production_points INTEGER,
                        training_points INTEGER,
                        quiz_scores INTEGER)''')

    # Insert sample data from Hope and Faith production into the database
    # Assuming some columns for weight deviation, humidity, temperature, and team name
    hope_production_data = hope_production[['BATCH no.', 'BATCH WEIGHT (kg) BEFORE COOKING', 'BATCH WEIGHT (kg) AFTER COOKING']].copy()
    hope_production_data['product'] = 'Hope'
    hope_production_data['weight_deviation'] = hope_production['BATCH WEIGHT (kg) BEFORE COOKING'] - hope_production['BATCH WEIGHT (kg) AFTER COOKING']
    hope_production_data['humidity'] = 65  # Dummy value for humidity
    hope_production_data['temperature'] = 5  # Dummy value for temperature
    hope_production_data['team'] = 'Team A'

    faith_production_data = faith_production[['BATCH no.', 'BATCH WEIGHT (kg) BEFORE COOKING', 'BATCH WEIGHT (kg) AFTER COOKING']].copy()
    faith_production_data['product'] = 'Faith'
    faith_production_data['weight_deviation'] = faith_production['BATCH WEIGHT (kg) BEFORE COOKING'] - faith_production['BATCH WEIGHT (kg) AFTER COOKING']
    faith_production_data['humidity'] = 70  # Dummy value for humidity
    faith_production_data['temperature'] = 6  # Dummy value for temperature
    faith_production_data['team'] = 'Team B'

    # Concatenate Hope and Faith data
    production_data = pd.concat([hope_production_data, faith_production_data], ignore_index=True)

    # Insert data into production_metrics table
    for _, row in production_data.iterrows():
        cursor.execute('''INSERT INTO production_metrics (product, weight_deviation, humidity, temperature, team)
                        VALUES (?, ?, ?, ?, ?)''',
                    (row['product'], row['weight_deviation'], row['humidity'], row['temperature'], row['team']))
        
    # Create production flows table
    cursor.execute('''CREATE TABLE IF NOT EXISTS production_flows (
                        id INTEGER PRIMARY KEY,
                        flow_name TEXT UNIQUE,
                        initial_points INTEGER,
                        current_points INTEGER)''')
        
    # Insert default production flow data if not present
    stages = [('Preproduction', 100), ('Cooking', 100), ('Storage', 100), ('Packing', 100)]
    for flow_name, points in stages:
        cursor.execute('''INSERT OR IGNORE INTO production_flows (flow_name, initial_points, current_points)
                          VALUES (?, ?, ?)''', (flow_name, points, points))

    # Create thresholds table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS thresholds (
                        id INTEGER PRIMARY KEY,
                        weight_deviation_threshold REAL,
                        humidity_threshold REAL,
                        temperature_threshold REAL)''')
    
    # Insert default threshold values if not present
    cursor.execute("INSERT OR IGNORE INTO thresholds (id, weight_deviation_threshold, humidity_threshold, temperature_threshold) VALUES (1, 2, 75, 6)")

    # Commit changes and close the database connection
    conn.commit()
    conn.close()

    print("Data inserted successfully into the database.")

# Initialize the database
init_db()

# Function to fetch data from SQLite and convert it to DataFrames
def fetch_data():
    conn = sqlite3.connect('dashboard.db')
    production_data = pd.read_sql_query("SELECT * FROM production_metrics", conn)
    team_scores = pd.read_sql_query("SELECT * FROM team_scores", conn)
    # Fetch thresholds and ensure they are scalars, not Series
    thresholds = pd.read_sql_query("SELECT * FROM thresholds", conn).iloc[0]
    weight_threshold = thresholds['weight_deviation_threshold']
    humidity_threshold = thresholds['humidity_threshold']
    temperature_threshold = thresholds['temperature_threshold']
    flows = pd.read_sql_query("SELECT * FROM production_flows", conn)

    # Check each record for deviations and adjust points accordingly
    for _, row in production_data.iterrows():
        if abs(row['weight_deviation']) > weight_threshold:
            # Deduct points for each stage based on failure
            flow_name = 'Cooking' if row['product'] == 'Hope' else 'Storage'
            conn.execute('''UPDATE production_flows
                            SET current_points = current_points - 5
                            WHERE flow_name = ? AND current_points > 0''', (flow_name,))
    
    # Fetch updated flows data after point adjustments
    updated_flows = pd.read_sql_query("SELECT * FROM production_flows", conn)
    conn.close()
    return production_data, updated_flows, team_scores, thresholds.iloc[0]

# Function to fetch data and train the predictive model
def train_predictive_model():
    conn = sqlite3.connect('dashboard.db')
    production_data = pd.read_sql_query("SELECT * FROM production_metrics", conn)
    conn.close()
    
    if len(production_data) < 10:
        return None, None  # Not enough data for meaningful predictions
    
    # Prepare data for model
    X = production_data[['humidity', 'temperature']].copy()
    y = production_data['weight_deviation']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions and evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    
    # Forecasting (simulate future data for display)
    future_humidity = np.random.uniform(60, 80, 5)
    future_temperature = np.random.uniform(5, 8, 5)
    future_data = pd.DataFrame({'humidity': future_humidity, 'temperature': future_temperature})
    forecasted_deviation = model.predict(future_data)
    
    forecast_df = pd.DataFrame({
        'Future Batch': range(1, 6),
        'Predicted Weight Deviation': forecasted_deviation
    })
    
    return forecast_df, mae

# Routes for dashboard display
@app.route('/')
def dashboard():
    # Fetch data and train model
    forecast_df, mae = train_predictive_model()
    production_data, updated_flows, team_scores, thresholds = fetch_data()

    # Points by Production Flow chart
    fig_flows = px.bar(updated_flows, x='flow_name', y='current_points', title="Current Points by Production Flow")
    flows_chart = pio.to_html(fig_flows, full_html=False)
    
    # Production chart
    fig_prod = px.bar(production_data, x='product', y='weight_deviation', color='team', title="Weight Deviations by Product")
    prod_chart = pio.to_html(fig_prod, full_html=False)
    
    # Team scores leaderboard
    fig_team = px.bar(team_scores, x='team', y='production_points', title="Team Scores Leaderboard")
    team_chart = pio.to_html(fig_team, full_html=False)
    
    # Prediction chart (if data is available)
    if forecast_df is not None:
        fig_pred = px.bar(forecast_df, x='Future Batch', y='Predicted Weight Deviation', title="Predicted Weight Deviations for Future Batches")
        pred_chart = pio.to_html(fig_pred, full_html=False)
    else:
        pred_chart = "<p>Not enough data for predictions</p>"

    return render_template('dashboard.html', prod_chart=prod_chart, team_chart=team_chart, pred_chart=pred_chart, flows_chart=flows_chart, thresholds=thresholds, mae=mae)

# Route to update thresholds (as in previous example)
@app.route('/update_thresholds', methods=['POST'])
def update_thresholds():
    weight_deviation_threshold = request.form['weight_deviation_threshold']
    humidity_threshold = request.form['humidity_threshold']
    temperature_threshold = request.form['temperature_threshold']
    
    conn = sqlite3.connect('dashboard.db')
    cursor = conn.cursor()
    cursor.execute('''UPDATE thresholds SET
                      weight_deviation_threshold = ?,
                      humidity_threshold = ?,
                      temperature_threshold = ?
                      WHERE id = 1''',
                   (weight_deviation_threshold, humidity_threshold, temperature_threshold))
    conn.commit()
    conn.close()
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)