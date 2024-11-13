# Insider Threat Detection: A Comparative Analysis Using Machine Learning

## Project Overview

This project focuses on the analysis and detection of insider threats in organizational data, employing a variety of machine learning techniques. Specifically, the research compares **regression models**, **supervised learning models**, and **unsupervised learning models** to identify anomalous behaviors indicative of insider threats.

The dataset consists of email communication logs, and the primary task is to identify anomalous behaviors in the data that could suggest insider threats, such as abnormal email sizes, recipient counts, and unusual hours of activity.

### Problem Statement

The goal is to develop a model that can identify patterns of anomalous behavior that may indicate insider threats, using a combination of **supervised** and **unsupervised** learning techniques. The models aim to distinguish between normal and suspicious activities based on extracted features, such as email size, recipient count, and time of day.

## Approach & Methodology

### Data Preprocessing & Feature Engineering

The data preparation phase involved multiple steps:

1. **Data Cleaning:**  
   Missing or inconsistent data was handled appropriately to ensure reliable model training.
   
2. **Exploratory Data Analysis (EDA):**  
   Performed an in-depth exploration of the dataset to understand distributions, correlations, and potential patterns in the data.
   
3. **Feature Extraction and Selection:**  
   Key features were extracted to represent email behavior and potential anomalies:
   - **Hour of the email:** Extracted from the `date` column to capture the time of day.
   - **Recipient count:** A derived feature that counts the total number of recipients (to, cc, bcc).
   - **Email size:** Captured from the `size` column to identify unusually large emails.

4. **Anomaly Detection Feature Engineering:**
   - **Size Z-score and Recipient Count Z-score:** Z-scores were calculated to identify outliers in the email size and recipient count.
   - **Threshold-based anomaly detection:** A combination of thresholding on email size, recipient count, and time of day (night hours) helped to mark potential anomalies.
   
   The anomaly detection feature is computed as follows:
   ```python
   df['hour'] = pd.to_datetime(df['date']).dt.hour
   df['recipient_count'] = df[['to', 'cc', 'bcc']].apply(lambda x: sum(pd.notnull(x)), axis=1)

   size_threshold = df['size'].quantile(0.95)  
   recipient_threshold = df['recipient_count'].quantile(0.95)  
   night_hours = [0, 1, 2, 3, 4, 5, 22, 23]  

   df['size_zscore'] = zscore(df['size'])
   df['recipient_count_zscore'] = zscore(df['recipient_count'])

   df['anomaly'] = (
       (df['size'] > size_threshold) |
       (df['recipient_count'] > recipient_threshold) |
       (df['hour'].isin(night_hours)) |
       (df['size_zscore'] > 3) |
       (df['recipient_count_zscore'] > 3)
   ).astype(int)
   ```

### Supervised Learning Models

The following **supervised learning models** were implemented to detect insider threats:

1. **Logistic Regression:** A simple linear model used to predict binary outcomes (normal or anomalous).
2. **Support Vector Machine (SVM):**
   - **Linear Kernel:** Applied to classify data using a linear decision boundary.
   - **RBF Kernel:** Used to capture more complex, non-linear relationships in the data.
3. **Ensemble Learning - Bagging:**
   - **K-Nearest Neighbors (KNN):** Combined with bagging to increase model robustness.
   - **Random Forest (RF):** Leveraged as part of an ensemble method to improve accuracy and reduce overfitting.

### Next Steps: Unsupervised Models

The next phase of the project involves applying **unsupervised models** to detect anomalies without the need for labeled data:

1. **Hierarchical Clustering:**  
   This model will help cluster the data based on similarity, providing insight into potential groups of normal and anomalous behavior.

2. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise):**  
   A density-based clustering algorithm that will be used to identify outliers and noise in the dataset, which could be indicative of insider threats.


## Requirements

To run the project, the following Python packages are required:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `scipy`

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Results & Analysis

After training the models, performance metrics such as **accuracy**, **precision**, **recall**, and **F1-score** are evaluated and shown at the end of each model execution. Additionally, the effectiveness of the unsupervised models in identifying anomalies will be assessed by analyzing clustering results and detecting outliers.

## Future Work

In addition to implementing the unsupervised models (Hierarchical and DBSCAN), the following steps are planned for the next phase of the project:

- Fine-tuning models and hyperparameters to improve performance.
- Testing additional anomaly detection techniques and comparing their effectiveness.
- Deploying a real-time anomaly detection system for continuous monitoring of organizational data.

