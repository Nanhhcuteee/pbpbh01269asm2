import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files from your local machine
customer_data = pd.read_csv('customer_data.csv')
sale_data = pd.read_csv('C:/Users/pbphu/Downloads/sale_data.csv')
product_detail_data = pd.read_csv('product_detail_data.csv')
product_group_data = pd.read_csv('product_group_data.csv')
market_trend_data = pd.read_csv('market_trend_data.csv')
website_access_data = pd.read_csv('website_access_data.csv')

# Example 1: Bar Chart - Total Amount by Product Group
product_sales = sale_data.merge(product_detail_data, on='product_id')
product_sales_grouped = product_sales.groupby('product_group_id')['total_amount'].sum().reset_index()
product_sales_grouped = product_sales_grouped.merge(product_group_data, on='product_group_id')

plt.figure(figsize=(10, 6))
plt.bar(product_sales_grouped['group_name'], product_sales_grouped['total_amount'])
plt.title('Total Sales Amount by Product Group')
plt.xlabel('Product Group')
plt.ylabel('Total Amount')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Example 2: Line Chart - Sales Over Time
sale_data['sale_date'] = pd.to_datetime(sale_data['sale_date'])
sales_over_time = sale_data.groupby(sale_data['sale_date'].dt.to_period('M'))['total_amount'].sum()

plt.figure(figsize=(10, 6))
sales_over_time.plot(kind='line')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.tight_layout()
plt.show()

# Example 3: Pie Chart - Market Trend Impact Distribution
market_trend_impact = market_trend_data['impact_level'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(market_trend_impact, labels=market_trend_impact.index, autopct='%1.1f%%', startangle=140)
plt.title('Market Trend Impact Distribution')
plt.tight_layout()
plt.show()

# Example 4: Scatter Plot - Relationship Between Discount and Total Amount
plt.figure(figsize=(10, 6))
plt.scatter(sale_data['discount'], sale_data['total_amount'])
plt.title('Relationship Between Discount and Total Amount')
plt.xlabel('Discount')
plt.ylabel('Total Amount')
plt.grid(True)
plt.tight_layout()
plt.show()

# Example 5: Donut Chart - Payment Method Distribution
payment_methods = sale_data['payment_method'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(payment_methods, labels=payment_methods.index, autopct='%1.1f%%', startangle=140, wedgeprops=dict(width=0.3))
plt.title('Payment Method Distribution')
plt.tight_layout()
plt.show()
