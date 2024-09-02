import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
file_path = 'PEP.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Plotting stock prices over time
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['Open'], label='Open Price')
plt.plot(data['High'], label='High Price')
plt.plot(data['Low'], label='Low Price')
plt.title('Stock Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Plotting trading volume over time
plt.figure(figsize=(14, 7))
plt.bar(data.index, data['Volume'], width=2)
plt.title('Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.show()
