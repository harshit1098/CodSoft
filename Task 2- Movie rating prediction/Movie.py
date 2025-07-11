# Movie Rating Prediction 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset with encoding fix
df = pd.read_csv("C:/Users/rajha/OneDrive/Desktop/Python/IMDb Movies India.csv", encoding='latin1')

# Clean column names (removes extra spaces)
df.columns = df.columns.str.strip()

# Drop rows with missing target values in 'Rating'
df.dropna(subset=['Rating'], inplace=True)

# Clean 'Votes' column: remove commas, convert to float
df['Votes'] = df['Votes'].astype(str).str.replace(',', '', regex=True).astype(float)

# Clean 'Duration' column: remove 'min', convert to float
df['Duration'] = df['Duration'].astype(str).str.replace('min', '', regex=True).str.strip().astype(float)

# Encode categorical columns
label_encoders = {}
for column in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Features and target
X = df[['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Votes', 'Year', 'Duration']]
y = df['Rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model using Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
