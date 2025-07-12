#Movie Rating Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("C:\\Users\\ADMIN\\Desktop\\movie rating prediction\\IMDb Movies India.csv", encoding='ISO-8859-1')

# Show first 5 rows
print("First 5 rows:\n", df.head())

# Show dataset info
print("\nDataset info:")
print(df.info())

# Drop missing ratings
df = df.dropna(subset=["Rating"])

# Create target label: 1 if Rating >= 7, else 0
df['Rating_Label'] = df['Rating'].apply(lambda x: 1 if x >= 7 else 0)

# Selected features for modeling
features = ['Year', 'Duration', 'Genre', 'Votes', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']
df = df[features + ['Rating_Label']].dropna()

# Encode categorical features
le = LabelEncoder()
for col in features:
    df[col] = le.fit_transform(df[col].astype(str))

# Feature matrix and target
X = df[features]
y = df['Rating_Label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
