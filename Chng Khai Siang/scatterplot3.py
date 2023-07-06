import pandas as pd
import matplotlib.pyplot as plt

# Load the electronic dataset
electronic_data = pd.read_csv('electronic_dataset.csv')

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

#MANIPULATING THE DATA
#Check for missing data
print(jewelry_data.isna().sum())
print(electronic_data.isna().sum())

#Remove missing data
jewelry_data = jewelry_data.dropna()
electronic_data = electronic_data.dropna()

# Preprocess the electronic data
electronic_data['event_time'] = pd.to_datetime(electronic_data['event_time'])
electronic_data['month'] = electronic_data['event_time'].dt.month
electronic_data['year'] = electronic_data['event_time'].dt.year

# Preprocess the jewelry data
jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
jewelry_data['month'] = jewelry_data['event_time'].dt.month
jewelry_data['year'] = jewelry_data['event_time'].dt.year

# Combine the datasets
combined_data = pd.concat([electronic_data, jewelry_data])

# Group the data by month, year, and category_code, and calculate the total number of purchases and the average price
grouped_data = combined_data.groupby(['year', 'month']).agg({'price': 'mean', 'order_id': 'count'}).reset_index()
grouped_data.rename(columns={'order_id': 'purchase_count'}, inplace=True)

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(grouped_data['price'], grouped_data['purchase_count'])
plt.xlabel('Price')
plt.ylabel('Purchase Count')
plt.title('Scatter Plot of Price vs Purchase Count')
plt.grid(True)
plt.show()
