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

# Group the electronic data by category_code and calculate the total number of purchases
electronic_grouped_data = electronic_data.groupby('category_code').size().reset_index(name='purchase_count_electronic')

# Group the jewelry data by category_code and calculate the total number of purchases
jewelry_grouped_data = jewelry_data.groupby('category_code').size().reset_index(name='purchase_count_jewelry')

# Select the top five categories from each dataset
top_categories_electronic = electronic_grouped_data.nlargest(5, 'purchase_count_electronic')['category_code'].tolist()
top_categories_jewelry = jewelry_grouped_data.nlargest(5, 'purchase_count_jewelry')['category_code'].tolist()

# Filter the electronic dataset to include only the top categories
filtered_electronic_data = electronic_data[electronic_data['category_code'].isin(top_categories_electronic)]

# Filter the jewelry dataset to include only the top categories
filtered_jewelry_data = jewelry_data[jewelry_data['category_code'].isin(top_categories_jewelry)]

# Concatenate the filtered datasets
combined_data = pd.concat([filtered_electronic_data, filtered_jewelry_data])

# Group the combined data by month, year, and category_code, and calculate the total number of purchases
grouped_data = combined_data.groupby(['year', 'month', 'category_code']).size().reset_index(name='purchase_count')

# Create the scatter plot
plt.figure(figsize=(10, 6))
for category in top_categories_electronic:
    category_data = grouped_data[grouped_data['category_code'] == category]
    plt.scatter(category_data['purchase_count'], category_data['category_code'], label=category)

for category in top_categories_jewelry:
    category_data = grouped_data[grouped_data['category_code'] == category]
    plt.scatter(category_data['purchase_count'], category_data['category_code'], label=category)

plt.xlabel('Purchase Count')
plt.ylabel('Category Code')
plt.title('Scatter Plot of Purchase Count vs Category Code')
plt.legend()
plt.grid(True)
plt.show()



