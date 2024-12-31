import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Set the matplotlib backend to 'Agg'
plt.switch_backend('Agg')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('IMDB Dataset.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Handle missing values, if any
print("Handling missing values...")
df = df.dropna()

# Handle duplicate values
print("Handling duplicate values...")
df = df.drop_duplicates()

# Convert categorical data to numerical (optional, if needed)
# No categorical data in this dataset, so this step is skipped.

# Feature engineering (if needed)
# No additional features are created in this example.

print("Data preparation finished.")

# Split the data into features and target variable
X = df['review']
y = df['sentiment']

# Convert target variable to numerical
print("Encoding target variable...")
y = y.map({'positive': 1, 'negative': 0})

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Analysis: Visualizations
print("Performing data analysis...")
sns.countplot(x='sentiment', data=df)
plt.title('Sentiment Distribution')
plt.savefig('sentiment_distribution.png')
print("Sentiment distribution plot saved as 'sentiment_distribution.png'.")

# Application of Machine Learning Techniques
print("Building and training the model...")
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train the model
text_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = text_clf.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
print("Saving the trained model...")
joblib.dump(text_clf, 'sentiment_model.pkl')

print("Model training finished.")