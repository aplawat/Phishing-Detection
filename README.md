# Phishing Detection System

A robust machine learning pipeline designed to fetch, process, and classify phishing URLs. This project leverages Apache Spark (PySpark) for large-scale data preprocessing and model training, Kafka for real-time data streaming, and Flask to expose the training pipeline as a RESTful API.

## Features

* **Automated Data Collection**: Fetches the latest phishing data from prominent threat intelligence feeds (PhishTank and OpenPhish).
* **Real-time & Batch Processing**: Supports batch fetching via HTTP requests and real-time data streaming via Apache Kafka.
* **Scalable Data Preprocessing**: Uses PySpark to handle large datasets, extracting key URL features such as domain information, special character counts, and URL lengths.
* **Machine Learning Pipeline**: Trains a Random Forest Classifier using PySpark MLlib, complete with feature indexing, encoding, and vector assembly.
* **REST API**: Exposes a Flask endpoint to trigger the end-to-end data fetching, processing, and training pipeline seamlessly.
* **Evaluation & Visualization**: Automatically generates comprehensive evaluation metrics (AUC, Precision, Recall, F1) and visual plots (ROC Curve, Confusion Matrix, Feature Importance).

## Project Structure

* `app.py`: The main Flask application providing the REST API endpoints (`/train` and `/health`).
* `collector.py`: Handles batch HTTP data extraction from PhishTank and OpenPhish.
* `data_collector.py`: A Kafka producer script that fetches threat feeds and streams them to a `phishing-urls` Kafka topic.
* `spark_processor.py`: PySpark module for data cleaning, URL parsing, and feature engineering.
* `train_model.py`: Handles the PySpark MLlib Random Forest model training, evaluation, saving the model (`phishing_rf_model/`), and generating visual performance charts.
* `requirements.txt`: Contains all necessary Python dependencies (Flask, requests, pyspark, kafka-python).
* `visualizations/`: Output directory where model evaluation plots (ROC, Confusion Matrix, etc.) are saved.

## Prerequisites

* Python 3.8+
* Apache Spark installed and configured on your system
* Apache Kafka (Optional: only required if you intend to run `data_collector.py` for real-time streaming)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Phishing-Detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Running the Flask API

You can start the Flask application to run the system as an API service.

```bash
python app.py
```

The server will start on `http://0.0.0.0:8000`.

**Trigger a Model Training Run:**
You can trigger the full pipeline (data collection -> spark preprocessing -> model training) via a POST request:

```bash
curl -X POST http://localhost:8000/train
```

**Check API Health:**

```bash
curl http://localhost:8000/health
```

### 2. Running Standalone Model Training and Evaluation

If you want to train the model directly, view evaluation metrics in the terminal, and generate visualizations without starting the API, run:

```bash
python train_model.py
```

This script will:

- Fetch and preprocess the data
- Train the Random Forest classifier
- Print out evaluation metrics (Accuracy, F1-Score, AUC-ROC)
- Save the trained model to the `./phishing_rf_model` directory
- Save evaluation charts into the `visualizations/` folder

### 3. Running the Kafka Data Collector

If you have a local Kafka broker running on localhost:9092 and want to stream phishing URLs continuously (fetches every hour):

```bash
python data_collector.py
```
## Model Details

The machine learning pipeline extracts the following features from raw URLs to determine if they are legitimate or malicious:

- **domain_vec**: One-Hot Encoded domain names.
- **special_char_count**: The frequency of suspicious special characters (@, ?, -, =, etc.).
- **url_length**: Total character length of the URL.
- **is_https**: Binary flag indicating the presence of a secure HTTPS connection.
- **Algorithm**: Random Forest Classifier (numTrees=100, maxDepth=8)

## Visualizations

When the model is trained, the following visualizations are automatically generated in the `visualizations/` directory:

- **Confusion Matrix** (confusion_matrix.png): Shows True Positives, True Negatives, False Positives, and False Negatives.
- **Feature Importance** (feature_importance.png): Ranks the engineered features by their impact on the model's decision-making.
- **ROC Curve** (roc_curve.png): Displays the diagnostic ability of the binary classifier system.
- **Prediction Distribution** (prediction_distribution.png): A histogram showing the distribution of probability scores across legitimate and phishing classes.