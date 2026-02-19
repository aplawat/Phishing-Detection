# train_model.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.sql.functions import col, expr
from pyspark.sql import SparkSession


def train_and_save_model(df):
    """Train a model on the processed data and save it"""
    
    # Split data into training and testing sets
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    
    # Define feature processing pipeline
    domain_indexer = StringIndexer(inputCol="domain", outputCol="domain_index", handleInvalid="keep")
    domain_encoder = OneHotEncoder(inputCol="domain_index", outputCol="domain_vec")
    
    # Assemble features into a single vector
    assembler = VectorAssembler(
        inputCols=["domain_vec", "special_char_count", "url_length", "is_https"], 
        outputCol="features"
    )
    
    # Create and configure the classifier
    classifier = RandomForestClassifier(
        featuresCol="features", 
        labelCol="label",
        numTrees=100,
        maxDepth=8,
        seed=42
    )
    
    # Create the pipeline
    pipeline = Pipeline(stages=[domain_indexer, domain_encoder, assembler, classifier])
    
    # Train the model
    model = pipeline.fit(train_data)
    
    # Evaluate the model
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(labelCol="label")
    auc = evaluator.evaluate(predictions)
    print(f"Model AUC: {auc}")
    
    # Save the model
    model_path = "./phishing_rf_model"
    model.write().overwrite().save(model_path)
    
    return model_path, model, predictions, test_data


def evaluate_model(predictions):
    """Perform detailed model evaluation and print metrics"""
    # Binary classification metrics
    binary_evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    auc_roc = binary_evaluator.evaluate(predictions)
    
    binary_evaluator.setMetricName("areaUnderPR")
    auc_pr = binary_evaluator.evaluate(predictions)
    
    # Multi-class metrics (even though we have binary classification)
    multi_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    
    accuracy = multi_evaluator.setMetricName("accuracy").evaluate(predictions)
    precision = multi_evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    recall = multi_evaluator.setMetricName("weightedRecall").evaluate(predictions)
    f1 = multi_evaluator.setMetricName("f1").evaluate(predictions)
    
    # Print the metrics
    print("\n======= Model Evaluation Metrics =======")
    print(f"Area under ROC: {auc_roc:.4f}")
    print(f"Area under PR: {auc_pr:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Create confusion matrix
    predictions_count = predictions.select("label", "prediction").groupBy("label", "prediction").count()
    confusion_matrix = predictions_count.toPandas().pivot(index='label', columns='prediction', values='count').fillna(0)
    
    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix
    }


def visualize_results(metrics, predictions, model, feature_cols):
    """Create and save visualizations for model evaluation"""
    # Create a directory for visualizations if it doesn't exist
    if not os.path.exists("visualizations"):
        os.makedirs("visualizations")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    if metrics["confusion_matrix"].shape == (2, 2):
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='g', cmap='Blues',
                    xticklabels=['Benign', 'Phishing'], yticklabels=['Benign', 'Phishing'])
    else:
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("visualizations/confusion_matrix.png")
    plt.close()
    
    # 2. ROC Curve
    # Convert predictions to pandas for easier plotting
    pred_pd = predictions.select("label", "prediction", "probability").toPandas()
    pred_pd["prob_phishing"] = pred_pd["probability"].apply(lambda x: float(x[1]))
    
    fpr, tpr, thresholds = calculate_roc(pred_pd["label"], pred_pd["prob_phishing"])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc_roc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig("visualizations/roc_curve.png")
    plt.close()
    
    # 3. Feature Importance
    rf_model = model.stages[-1]
    feature_importances = rf_model.featureImportances.toArray()
    
    # This is a simplified approach since we have a combined vector
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importances)), feature_importances)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.grid(True, alpha=0.3)
    plt.savefig("visualizations/feature_importance.png")
    plt.close()
    
    # 4. Prediction Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pred_pd, x="prob_phishing", hue="label", bins=30, alpha=0.7)
    plt.title('Distribution of Prediction Probabilities by Class')
    plt.xlabel('Probability of Phishing')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig("visualizations/prediction_distribution.png")
    plt.close()
    
    print("\nVisualizations saved in the 'visualizations' directory.")


def calculate_roc(y_true, y_score):
    """Calculate ROC curve points"""
    from sklearn.metrics import roc_curve
    return roc_curve(y_true, y_score)


def analyze_misclassifications(predictions):
    """Analyze and report on misclassified instances"""
    # Filter for misclassifications
    misclassified = predictions.filter(col("label") != col("prediction"))
    
    # Count by type of error
    false_positives = misclassified.filter(col("label") == 0).count()
    false_negatives = misclassified.filter(col("label") == 1).count()
    
    print("\n======= Misclassification Analysis =======")
    print(f"Total misclassifications: {misclassified.count()}")
    print(f"False positives (benign URLs classified as phishing): {false_positives}")
    print(f"False negatives (phishing URLs missed): {false_negatives}")
    
    # Sample misclassified instances
    print("\nSample misclassified instances:")
    misclassified.select("url", "domain", "label", "prediction", "probability").limit(5).show(truncate=False)
    
    return misclassified


def get_feature_columns(df, model):
    """Get the list of feature column names"""
    # This is a simplified approach - in a real scenario, you'd extract this from your pipeline
    feature_cols = ["domain", "special_char_count", "url_length", "is_https"]
    return feature_cols


def main():
    """Main function to demonstrate model training, evaluation and visualization"""
    print("=== Phishing Detection Model Training and Evaluation ===\n")
    
    # Initialize Spark
    spark = SparkSession.builder.appName("PhishingModelEvaluation").getOrCreate()
    
    # Check if we have sample data or need to generate it
    try:
        # Try to load processed data if available
        from collector import fetch_data
        from spark_processor import preprocess_data
        
        print("Fetching and processing data...")
        raw_data = fetch_data()
        df = preprocess_data(raw_data)
        print(f"Processed {df.count()} records")
        
    except Exception as e:
        print(f"Could not load or process real data: {e}")
        print("Creating sample data instead...")
        
        # Create sample data for demonstration
        data = [
            {"url": "https://legitimate-bank.com/login", "domain": "legitimate-bank.com", 
             "special_char_count": 2, "url_length": 30, "is_https": 1, "label": 0},
            {"url": "https://phishing-site.xyz/login/bank", "domain": "phishing-site.xyz", 
             "special_char_count": 3, "url_length": 35, "is_https": 1, "label": 1},
            # Add more examples...
        ]
        
        # Generate more synthetic data
        for i in range(1000):
            # Legitimate URLs
            if i % 2 == 0:
                legitimate = {
                    "url": f"https://legitimate-site-{i}.com/page{i}", 
                    "domain": f"legitimate-site-{i}.com",
                    "special_char_count": np.random.randint(1, 5),
                    "url_length": np.random.randint(20, 40),
                    "is_https": 1,
                    "label": 0
                }
                data.append(legitimate)
            # Phishing URLs
            else:
                phishing = {
                    "url": f"https://bank-secure-login-{i}.ru/secure/login{i}.php", 
                    "domain": f"bank-secure-login-{i}.ru",
                    "special_char_count": np.random.randint(4, 15),
                    "url_length": np.random.randint(35, 70),
                    "is_https": np.random.randint(0, 2),
                    "label": 1
                }
                data.append(phishing)
                
        df = spark.createDataFrame(data)
        print(f"Created {df.count()} synthetic records for demonstration")
    
    # Show data distribution
    print("\nData distribution:")
    df.groupBy("label").count().show()
    
    # Train and evaluate the model
    print("\nTraining model...")
    model_path, model, predictions, test_data = train_and_save_model(df)
    print(f"Model saved to: {model_path}")
    
    # Detailed evaluation
    metrics = evaluate_model(predictions)
    
    # Get feature columns
    feature_cols = get_feature_columns(df, model)
    
    # Create visualizations
    visualize_results(metrics, predictions, model, feature_cols)
    
    # Analyze misclassifications
    misclassified = analyze_misclassifications(predictions)
    
    print("\n=== Model Training and Evaluation Complete ===")
    
    # Optional: Demonstrate model on new URLs
    print("\nDemonstrating model on new sample URLs:")
    
    # Create test cases
    test_cases = [
        {"url": "https://www.google.com", "domain": "www.google.com", 
         "special_char_count": 2, "url_length": 22, "is_https": 1},
        {"url": "http://amaz0n-secure-login.xyz/account", "domain": "amaz0n-secure-login.xyz", 
         "special_char_count": 8, "url_length": 40, "is_https": 0}
    ]
    
    test_df = spark.createDataFrame(test_cases)
    test_predictions = model.transform(test_df)
    
    # Show results
    results = test_predictions.select("url", "prediction", "probability").collect()
    for row in results:
        url = row["url"]
        prediction = "Phishing" if row["prediction"] == 1.0 else "Legitimate"
        probability = row["probability"][1]  # Probability of being phishing
        print(f"URL: {url}")
        print(f"Prediction: {prediction} (Confidence: {probability:.4f})")
        print("---")
    
    # Clean up
    spark.stop()


if __name__ == "__main__":
    main()