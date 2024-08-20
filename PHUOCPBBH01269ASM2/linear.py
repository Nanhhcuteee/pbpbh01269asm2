import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load the sales data from the CSV file
data = pd.read_csv('sale_data.csv')  # Replace with the correct path

# Prepare the data by selecting relevant features
X = data[['discount', 'quantity']]  # Independent variables
y = data['total_amount']  # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error for model evaluation
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Visualize the relationship between actual and predicted sales with a regression line
plt.scatter(y_test, y_pred, color='blue')

# Add the regression line (line of best fit)
# We need to fit a line on the scatter plot, so we take the min and max of y_test
line = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(line)
plt.plot(y_test, p(y_test), color='red')

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales with Regression Line")
plt.show()

# Predict future sales for a given discount and quantity
future_sales_prediction = model.predict([[30, 5]])  # Example: 30% discount and quantity of 5
print(f'Predicted Future Sales: {future_sales_prediction}')
