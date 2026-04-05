```markdown
# Explainable AI (XAI) for Medical Imaging & Health Monitoring

## Project Overview
This project demonstrates various applications of Artificial Intelligence (AI) in healthcare, focusing on both diagnostic assistance and personal health monitoring. It also provides a comprehensive suite of machine learning model performance visualizations, generated using synthetic data, to illustrate key evaluation metrics.

## Features
-   **Medical Imaging Diagnosis (XAI)**: Utilizes Grad-CAM to provide explainable insights into medical image classifications (e.g., X-rays).
-   **Daily Health Assistant**: A rule-based AI system for real-time health status monitoring based on vital signs.
-   **Machine Learning Model Performance Visualizations**: A collection of plots to evaluate and understand classification model behavior.

## Visualizations Included

### 1. Original Medical Scan & XAI Explanation (Heatmap)
*   **Purpose**: Shows the input medical image alongside a Grad-CAM heatmap, highlighting regions of interest that the AI model focused on during diagnosis.
*   **Insight**: Provides transparency into the model's decision-making process for image-based predictions.

### 2. Classification Report Heatmap
*   **Purpose**: Visualizes precision, recall, and F1-score for each class in a multi-class classification scenario.
*   **Insight**: Offers a quick overview of model performance across different categories, identifying classes where the model excels or struggles.

### 3. Confusion Matrix
*   **Purpose**: Displays the count of correct and incorrect predictions made by a classification model, broken down by class.
*   **Insight**: Helps to understand specific types of errors the model is making (e.g., confusing one class for another).

### 4. ROC Curve (Receiver Operating Characteristic)
*   **Purpose**: Plots the True Positive Rate against the False Positive Rate at various threshold settings for a classifier.
*   **Insight**: Evaluates the diagnostic ability of a binary or multi-class (One-vs-Rest) classifier across all possible thresholds, particularly useful for understanding sensitivity and specificity.

### 5. Precision-Recall Curve
*   **Purpose**: Illustrates the trade-off between precision and recall at different classification thresholds.
*   **Insight**: Especially valuable for imbalanced datasets, it shows how many of the positive predictions were correct (precision) and how many actual positives were caught (recall).

### 6. Calibration Plot (Reliability Diagram)
*   **Purpose**: Assesses how well a probabilistic classification model's predicted probabilities match the true event frequencies.
*   **Insight**: A perfectly calibrated model's predicted probabilities should match the fraction of positives, indicating reliable confidence scores.

### 7. Predicted Probability Distribution
*   **Purpose**: Histograms showing the distribution of predicted probabilities for each class.
*   **Insight**: Helps understand the model's confidence levels and whether it makes clear distinctions between classes or if probabilities are clustered.

### 8. Model Comparison Graphs
*   **Purpose**: Bar charts comparing different models based on their Accuracy and Explanation Quality Metrics (Localization Accuracy, Faithfulness, Interpretability Score).
*   **Insight**: Provides a visual comparison of various model performances and their explainability aspects.

## Setup and How to Run

1.  **Environment**: This project is designed to run in a Google Colab environment.
2.  **Dependencies**: The notebook includes `!pip install tf-explain -q` for necessary library installation.
    *   `tensorflow`
    *   `tf_explain`
    *   `matplotlib`
    *   `numpy`
    *   `opencv`
    *   `pandas`
    *   `scikit-learn`
    *   `seaborn`
3.  **Execution**: Simply run all cells in the Colab notebook sequentially.

## Usage

### Medical Imaging Diagnosis
The `run_medical_diagnosis(img_url)` function takes an image URL (e.g., a chest X-ray) and performs a diagnosis using a pre-trained VGG16 model, visualizing the regions of interest with Grad-CAM.

### Daily Health Assistant
The `daily_health_assistant(heart_rate, temp)` function provides a health status report and explanation based on input heart rate and temperature.

### Model Performance Visualizations
Synthetic data is generated within the notebook to demonstrate the classification report, confusion matrix, ROC curve, precision-recall curve, calibration plot, and predicted probability distribution graphs.

## Output
All generated plots are displayed inline in the notebook and saved as `.png` files (e.g., `calibration_plot.png`, `confusion_matrix.png`).
