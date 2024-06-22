import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

ticker = 'AAPL'

data = yf.download(ticker, start='2005-01-01', end='2024-01-11')
print(data.head())

data = data.dropna()
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=10).mean()
data = data.dropna()
X = data[['SMA_10', 'SMA_50']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='Close Price')
plt.plot(data.index, data['SMA_10'], label='10-Day SMA')
plt.plot(data.index, data['SMA_50'], label='50-Day SMA')
plt.legend()
plt.title('Stock Price and Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

new_data = [[data['SMA_10'].iloc[-1], data['SMA_50'].iloc[-1]]]
predicted_price = model.predict(new_data)
print(f'Predicted Stock Price: {predicted_price[0]}')

