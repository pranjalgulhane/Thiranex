import pandas as pd
import matplotlib.pyplot as plt
plt.show()
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Load data
data = pd.read_csv("sales_data.csv")
data['Date'] = pd.to_datetime(data['Date'])

# Plot data
plt.plot(data['Date'], data['Sales'])
plt.title("Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Prepare data
df = data[['Date', 'Sales']]
df.columns = ['ds', 'y']

# Train model
model = Prophet()
model.fit(df)

# Predict future
future = model.make_future_dataframe(periods=6, freq='M')
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.show()

# Evaluate model
predicted = forecast['yhat'][:len(df)]
mae = mean_absolute_error(df['y'], predicted)

print("Mean Absolute Error:", mae)

# Show predictions
print(forecast[['ds', 'yhat']].tail())