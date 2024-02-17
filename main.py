from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: Collect Data
stock_data = yf.download('AAPL', start='2010-01-01', end='2024-01-01')

# Step 2: Preprocess Data
stock_data['Next_Close'] = stock_data['Close'].shift(-1)  # Shift close price up by one day
stock_data.dropna(inplace=True)  # Drop rows with missing values

# Step 3: Split Data
X = stock_data[['Open', 'High', 'Low', 'Volume']]
y = stock_data['Next_Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Choose Model
model = LinearRegression()

# Step 5: Train Model
model.fit(X_train, y_train)

# Step 6: Evaluate Model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))

print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)

# Step 7: Make Predictions
latest_data = np.array([[stock_data['Open'][-1], stock_data['High'][-1], stock_data['Low'][-1], stock_data['Volume'][-1]]])
predicted_price = model.predict(latest_data)

# Get the prediction date
latest_date = stock_data.index[-1].date()
next_date = latest_date + timedelta(days=1)

# Get the day of the week for the prediction date
day_of_week = next_date.strftime('%A')

print("Predicted Next Close Price on", next_date, "({}):".format(day_of_week), predicted_price)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(stock_data.index, stock_data['Next_Close'], color='blue', label='Actual Close Price')
plt.plot(stock_data.index, model.predict(X), color='red', linestyle='-', label='Linear Regression')

# Plotting the predicted point
plt.scatter(next_date, predicted_price, color='green', label='Predicted Close Price')

plt.title('Linear Regression for AAPL Stock')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
