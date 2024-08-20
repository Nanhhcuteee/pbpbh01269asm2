import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the sales data from the CSV file
data = pd.read_csv('sale_data.csv')  # Thay thế đường dẫn chính xác

# Convert the sale_date column to datetime format
data['sale_date'] = pd.to_datetime(data['sale_date'])

# Aggregate sales data by date (ensure only numeric columns are summed)
sales_over_time = data.groupby(data['sale_date'].dt.to_period('M')).sum(numeric_only=True)['total_amount']

# Reset the index to get it in the right format for regression
sales_over_time = sales_over_time.reset_index()

# Prepare the data for modeling
# Convert the date to a numeric format (e.g., number of months since the start)
sales_over_time['sale_date'] = sales_over_time['sale_date'].apply(lambda x: x.to_timestamp())
sales_over_time['time_index'] = np.arange(len(sales_over_time))  # Numeric time index for regression

# Independent variable (time) and dependent variable (total sales amount)
X = sales_over_time[['time_index']]
y = sales_over_time['total_amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Predict future sales (extending the time index beyond the existing data)
future_time_index = np.arange(len(sales_over_time), len(sales_over_time) + 12)  # Predict for the next 12 months
future_sales = model.predict(future_time_index.reshape(-1, 1))

# Plot the actual sales data
plt.plot(sales_over_time['sale_date'], y, label='Actual Sales')

# Plot the predicted future sales
future_dates = pd.date_range(start=sales_over_time['sale_date'].iloc[-1], periods=12, freq='M')
plt.plot(future_dates, future_sales, label='Predicted Sales', linestyle='--', color='red')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.title('Sales Trends Over Time with Future Predictions')
plt.legend()
plt.grid(True)
plt.show()
