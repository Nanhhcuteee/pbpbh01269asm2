import pandas as pd

# Load the CSV files
customer_data = pd.read_csv('customer_data.csv')
sale_data = pd.read_csv('sale_data.csv')
product_detail_data = pd.read_csv('product_detail_data.csv')
product_group_data = pd.read_csv('product_group_data.csv')
market_trend_data = pd.read_csv('market_trend_data.csv')
website_access_data = pd.read_csv('website_access_data.csv')

# Handle missing values
customer_data.ffill(inplace=True)
sale_data.dropna(subset=['total_amount'], inplace=True)

# Correct data types
sale_data['sale_date'] = pd.to_datetime(sale_data['sale_date'])

# Remove duplicates
product_detail_data.drop_duplicates(inplace=True)

# Format data
market_trend_data['created_date'] = pd.to_datetime(market_trend_data['created_date'])

# Filter outliers (example: remove rows with negative total_amount)
sale_data = sale_data[sale_data['total_amount'] >= 0]

# Save the cleaned data (optional)
customer_data.to_csv('cleaned_customer_data.csv', index=False)
