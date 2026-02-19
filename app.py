# app.py
from flask import Flask, request, jsonify
from collector import fetch_data
from spark_processor import preprocess_data
from train_model import train_and_save_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    try:
        # Step 1: Fetch data
        logger.info("Fetching phishing data from sources...")
        raw_data = fetch_data()
        
        if not raw_data:
            return jsonify({'error': 'No data fetched from sources'}), 500
            
        logger.info(f"Successfully fetched {len(raw_data)} entries")
        
        # Step 2: Preprocess using Spark
        logger.info("Preprocessing data with Spark...")
        processed_data = preprocess_data(raw_data)  # Ensure this function is correctly written
        logger.info(f"Data processed. Shape: {processed_data.count()} rows")
        
        # Step 3: Train model
        logger.info("Training model...")
        model_path = train_and_save_model(processed_data)
        logger.info(f"Model trained and saved to {model_path}")

        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully', 
            'model_path': model_path,
            'dataset_size': processed_data.count()
        })
    except Exception as e:
        logger.error(f"Error in training process: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
