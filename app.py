from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_file, send_from_directory
import pandas as pd
import numpy as np
import io
import base64
import os
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename  # Secure filename handling
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html', title="Home", 
                          subtitle="A simple CSV interpreter. Gain basic metrics and visuals with a single upload.")

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles CSV Upload"""
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    
    if file.filename == '':
        return redirect(url_for('index'))

    if file and file.filename.endswith('.csv'):
        try:
            df = pd.read_csv(file)
            
            session['filename'] = secure_filename(file.filename)  # Secure filename
            session['columns'] = df.columns.tolist()
            session['row_count'] = len(df)

            # Save the dataframe
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{session['filename']}")
            df.to_csv(filepath, index=False)
            session['filepath'] = filepath  # Save correct path
            
            return redirect(url_for('upload_success'))
        except Exception as e:
            return f"Error processing CSV: {str(e)}"

    return redirect(url_for('index'))

@app.route('/upload_success')
def upload_success():
    if 'filename' not in session:
        return redirect(url_for('index'))
    
    return render_template('upload.html', title="Upload Success",
                          subtitle=f"Your CSV file '{session['filename']}' has been uploaded. Choose what you want to do next:",
                          filename=session['filename'],
                          row_count=session['row_count'])

@app.route('/eda')
def eda():
    """Perform Exploratory Data Analysis (EDA)"""
    if 'filepath' not in session:
        return redirect(url_for('index'))

    try:
        df = pd.read_csv(session['filepath'])

        info = {
            'filename': session['filename'],
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': df.columns.tolist()
        }

        columns = df.columns.tolist()
        numeric_summary = df.describe().to_html(classes='table bg-white text-black')
        dtypes = df.dtypes.astype(str).to_dict()
        missing_values = df.isnull().sum().to_dict()
        preview = df.head(10).to_html(classes='table bg-white text-black')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlation_plot = None

        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
            plt.title('Correlation Heatmap')

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            correlation_plot = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()

        return render_template('eda.html', 
                               title="Exploratory Data Analysis",
                               subtitle=f"Exploring the CSV file: {session['filename']}",
                               info=info,
                               preview=preview,
                               numeric_summary=numeric_summary,
                               dtypes=dtypes,
                               missing_values=missing_values,
                               correlation_plot=correlation_plot,
                               has_numeric_columns=(len(numeric_cols) > 0),
                               columns=columns)
    except Exception as e:
        return f"Error analyzing data: {str(e)}"

@app.route('/clean_data', methods=['POST'])
def clean_data():
    """Cleans the CSV file based on the selected missing value strategy."""
    if 'filepath' not in session:
        return jsonify({'error': 'No data available. Please upload a CSV file first.'})

    try:
        df = pd.read_csv(session['filepath'])
        data = request.get_json()
        strategy = data.get('missing_value_strategy', 'mean')

        if strategy == "mean":
            df.fillna(df.select_dtypes(include=[np.number]).mean(), inplace=True)  # Fix mean issue
        elif strategy == "first":
            df.fillna(method='ffill', inplace=True)
        elif strategy == "last":
            df.fillna(method='bfill', inplace=True)
        elif strategy == "delete":
            df.dropna(inplace=True)

        df.drop_duplicates(inplace=True)

        cleaned_filename = "cleaned_" + session['filename']
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)
        session['cleaned_filepath'] = cleaned_filepath  # Ensure filepath is stored

        return jsonify({'message': 'Data cleaned successfully!', 'download_url': f'/download_cleaned'})

    except Exception as e:
        return jsonify({'error': f'Error cleaning data: {str(e)}'})

@app.route('/download_cleaned')
def download_cleaned():
    """Allows users to download the cleaned CSV file."""
    if 'cleaned_filepath' not in session:
        return jsonify({'error': 'Cleaned data not found. Please clean the data first.'})

    cleaned_filepath = session['cleaned_filepath']

    if not os.path.exists(cleaned_filepath):
        return jsonify({'error': 'Cleaned file not found. Please clean the data first.'})

    return send_file(cleaned_filepath, as_attachment=True)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/graph')
def graph():
    if 'filepath' not in session:
        return redirect(url_for('index'))
    
    try:
        # Read the saved dataframe
        df = pd.read_csv(session['filepath'])
        
        # Get columns by type
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        return render_template('graph.html', 
                              title="Visualization",
                              subtitle=f"Create visualizations for: {session['filename']}",
                              numeric_cols=numeric_cols,
                              categorical_cols=categorical_cols)
    
    except Exception as e:
        return f"Error loading data for visualization: {str(e)}"

@app.route('/generate_graph', methods=['POST'])
def generate_graph():
    if 'filepath' not in session:
        return jsonify({'error': 'No data available. Please upload a CSV file first.'})
    
    try:
        data = request.get_json()
        chart_type = data.get('chart_type', 'bar')
        x_col = data.get('x_col')
        y_col = data.get('y_col')
        
        if not x_col or not y_col:
            return jsonify({'error': 'Please select both X and Y columns'})
        
        # Read the saved dataframe
        df = pd.read_csv(session['filepath'])
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == 'bar':
            plt.bar(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Bar Chart: {y_col} by {x_col}')
            plt.xticks(rotation=45)
            
        elif chart_type == 'line':
            plt.plot(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Line Chart: {y_col} by {x_col}')
            plt.xticks(rotation=45)
            
        elif chart_type == 'scatter':
            plt.scatter(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'Scatter Plot: {y_col} vs {x_col}')
            
        elif chart_type == 'pie':
            # For pie charts, limit to a manageable number of categories
            if df[x_col].nunique() > 10:
                top_values = df.groupby(x_col)[y_col].sum().nlargest(10)
                plt.pie(top_values, labels=top_values.index, autopct='%1.1f%%')
                plt.title(f'Pie Chart (Top 10): {y_col} by {x_col}')
            else:
                grouped = df.groupby(x_col)[y_col].sum()
                plt.pie(grouped, labels=grouped.index, autopct='%1.1f%%')
                plt.title(f'Pie Chart: {y_col} by {x_col}')
                
        elif chart_type == 'hist':
            plt.hist(df[y_col], bins=20)
            plt.xlabel(y_col)
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {y_col}')
            
        plt.tight_layout()
        
        # Save plot to a base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return jsonify({'image': img_str})
    
    except Exception as e:
        return jsonify({'error': f'Error generating graph: {str(e)}'})
if __name__ == '__main__':
    app.run(debug=True)
