import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

# Step 1: Input dataset (CSV file)
file_path = input("Enter the path to your stock data CSV file: ")
df = pd.read_csv(file_path)

# Basic checks—adjust these if your column names differ!
print("Columns in your dataset:", df.columns)
date_col = input("Enter the column name for date/time: ")
price_col = input("Enter the column name for the closing price: ")

# Convert date column to datetime and sort if needed
df[date_col] = pd.to_datetime(df[date_col])
df.sort_values(by=date_col, inplace=True)

# Step 2: Calculate Moving Averages (30 and 90 days as example)
df['MA30'] = df[price_col].rolling(window=30).mean()
df['MA90'] = df[price_col].rolling(window=90).mean()

# Step 3: Plot Close Price and Moving Averages
plt.figure(figsize=(14,7))
plt.plot(df[date_col], df[price_col], label="Close Price")
plt.plot(df[date_col], df['MA30'], label="30-day MA")
plt.plot(df[date_col], df['MA90'], label="90-day MA")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Prices with Moving Averages")
plt.legend()
plt.show()

# Step 4: Predict Trend Using Linear Regression
# Use recent N days for regression, e.g., last 120 data points
N = 120
recent = df.tail(N)

# Reshape the date for regression
recent['ordinal_date'] = recent[date_col].map(datetime.toordinal)
X = recent['ordinal_date'].to_numpy().reshape(-1, 1)
y = recent[price_col].to_numpy()

reg = LinearRegression()
reg.fit(X, y)
trend = reg.predict(X)

# Step 5: Plot Regression Line
plt.figure(figsize=(14,7))
plt.plot(recent[date_col], y, label="Actual Price")
plt.plot(recent[date_col], trend, label="Regression Trend", color='red')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Stock Price Trend (Regression on last {} days)".format(N))
plt.legend()
plt.show()#C:\Users\tomch\Downloads\ai2\UNIONBANK.csv
