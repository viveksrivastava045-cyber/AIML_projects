# Import libraries
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("PROJECT3- CANADA per capita income/canada_per_capita_income.csv")

# Display first five rows
print(df.head())

# Plot the data
plt.scatter(df['year'], df['per capita income (US$)'], color='red', marker='+')
plt.xlabel("Year")
plt.ylabel("Per Capita Income (US$)")
plt.show()

# Prepare training data
X = df[['year']]
y = df['per capita income (US$)']

# Create and train the model
model = LinearRegression()
model.fit(X, y)

# Predict income for the year 2020
prediction = model.predict([[2020]])

print("Predicted per capita income in 2020:")
print(prediction[0])

# Optional: Model parameters
print("\nSlope (Coefficient):", model.coef_[0])
print("Intercept:", model.intercept_)
