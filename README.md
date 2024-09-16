- 👋 import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv(r"C:\Users\harini p\Downloads\quikr_car.csv")

# Print column names and first few rows to verify
print("Column names:", df.columns)
print("First few rows of the dataset:")
print(df.head())

# Strip any leading or trailing spaces in column names
df.columns = df.columns.str.strip()

# Check for missing values
print("Missing values in each column:")
print(df.isna().sum())

# Handle missing values
df = df.dropna()

# Handle non-numeric values in 'Price' column
# First, identify non-numeric values
non_numeric_prices = df[~df['Price'].str.replace(',', '').str.replace('₹', '').str.isnumeric()]
print("Non-numeric values in 'Price' column:")
print(non_numeric_prices)

# Replace non-numeric values with NaN and then drop these rows
df['Price'] = pd.to_numeric(df['Price'].str.replace(',', '').str.replace('₹', ''), errors='coerce')
df = df.dropna(subset=['Price'])

# Convert 'kms_driven' from string to integer
df['kms_driven'] = df['kms_driven'].str.replace(' kms', '').str.replace(',', '').astype(float).astype(int)

# Drop the 'name' and 'company' columns as they are not used in the model
df = df.drop(columns=['name', 'company'])

# Define features and target variable
X = df[['year', 'kms_driven', 'fuel_type']]
y = df['Price']

# Convert categorical data into numeric using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['year', 'kms_driven']),
        ('cat', OneHotEncoder(), ['fuel_type'])
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict the price for a new car
new_car = pd.DataFrame({
    'year': [2020],  # Example: year of manufacture
    'kms_driven': [30000],  # Example: kilometers driven
    'fuel_type': ['Petrol']  # Example: fuel type
})

predicted_price = model.predict(new_car)
print(f"Predicted Price for the new car: ₹{predicted_price[0]:,.2f}")Hi, I’m @garishmaks08
- 👀 I’m interested in ...
- 🌱 I’m currently learning ...
- 💞️ I’m looking to collaborate on ...
- 📫 How to reach me ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...

<!---
garishmaks08/garishmaks08 is a ✨ special ✨ repository because its `README.md` (this file) appears on your GitHub profile.
You can click the Preview link to take a look at your changes.
--->
