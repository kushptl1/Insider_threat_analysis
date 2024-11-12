# Importing Dataset and doing the Exploratory Data Analysis. 

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras.layers import Dense, GRU
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression


# Load the dataset
df = pd.read_csv("/content/email.csv")

# Convert date field to datetime, if not already done
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Basic dataset overview
print(df.info())
print(df.describe())
print(df.head())

# 1. Distribution of Emails Sent by Hour
df['hour'] = df['date'].dt.hour
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='hour', bins=24, kde=True)
plt.title('Distribution of Emails Sent by Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Email Count')
plt.show()

# 2. Daily Email Frequency per User
daily_emails = df.groupby([df['date'].dt.date, 'user']).size().reset_index(name='email_count')
plt.figure(figsize=(12, 6))
sns.histplot(data=daily_emails, x='email_count', bins=30, kde=True)
plt.title('Daily Email Frequency per User')
plt.xlabel('Number of Emails per Day')
plt.ylabel('Frequency')
plt.show()

# 3. Average Email Size Distribution (in Bytes)
plt.figure(figsize=(12, 6))
sns.histplot(df['size'].dropna(), bins=50, kde=True)
plt.title('Distribution of Email Size')
plt.xlabel('Email Size (Bytes)')
plt.ylabel('Frequency')
plt.show()

# 4. Count of Recipients per Email
df['recipient_count'] = df[['to', 'cc', 'bcc']].apply(lambda x: sum(pd.notnull(x)), axis=1)
plt.figure(figsize=(12, 6))
sns.histplot(df['recipient_count'], bins=10, kde=True)
plt.title('Distribution of Recipient Count per Email')
plt.xlabel('Number of Recipients')
plt.ylabel('Frequency')
plt.show()

# 5. Heatmap of Email Activity by Day and Hour
email_counts = df.groupby([df['date'].dt.date, df['date'].dt.hour]).size().unstack(fill_value=0)
plt.figure(figsize=(14, 8))
sns.heatmap(email_counts, cmap="YlGnBu", cbar=True)
plt.title('Heatmap of Email Activity by Day and Hour')
plt.xlabel('Hour of the Day')
plt.ylabel('Date')
plt.show()

# Data Cleaning and feature Engineering

missing_values = data.isnull().sum()

# Basic statistics and info
data_info = data.describe()
data_info = data.info()

# Exploratory Data Analysis
email_volume = data.groupby(data['date'].dt.date).size()

# Frequency of emails by user
user_frequency = data['user'].value_counts()

# Number of attachments analysis
attachment_analysis = data['attachments'].value_counts()

# Display results
print("Missing Values:\n", missing_values)

print("Data Info:\n", data_info)

print("Email Volume Over Time:\n", email_volume)

print("User Frequency:\n", user_frequency)

print("Attachment Analysis:\n", attachment_analysis)

# defining the new feature which would be the target feature - Anamoly

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
    (df['size_zscore'] > 3) |  # Anomalous if size z-score is above 3
    (df['recipient_count_zscore'] > 3)  # Anomalous if recipient count z-score is above 3
).astype(int)

##############################################################################################

# Model #1: Logistic Regression:

X = df[['size', 'recipient_count', 'hour', 'attachments']]  # Use relevant features
y = df['anomaly']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)

y_pred = lr_model.predict(X_test_scaled)

print("Logistic Regression Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False)
    plt.title('Confusion Matrix for Logistic Regression')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(cm)
