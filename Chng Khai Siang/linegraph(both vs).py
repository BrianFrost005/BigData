import pandas as pd
import matplotlib.pyplot as plt

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Load the electronics dataset
electronics_data = pd.read_csv('electronic_dataset.csv')

# Convert event_time to datetime for jewelry dataset
jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])

# Convert event_time to datetime for electronics dataset
electronics_data['event_time'] = pd.to_datetime(electronics_data['event_time'])

# Filter the data for the year 2020 for jewelry dataset
jewelry_data = jewelry_data[jewelry_data['event_time'].dt.year == 2020]

# Filter the data for the year 2020 for electronics dataset
electronics_data = electronics_data[electronics_data['event_time'].dt.year == 2020]

# Filter out missing values and negative prices for jewelry dataset
jewelry_data = jewelry_data.dropna(subset=['price'])
jewelry_data = jewelry_data[jewelry_data['price'] > 0]

# Filter out missing values and negative prices for electronics dataset
electronics_data = electronics_data.dropna(subset=['price'])
electronics_data = electronics_data[electronics_data['price'] > 0]

# Calculate the revenue of jewelry for each month in 2020
jewelry_revenue_by_month = jewelry_data.groupby(jewelry_data['event_time'].dt.month)['price'].sum()

# Calculate the revenue of electronics for each month in 2020
electronics_revenue_by_month = electronics_data.groupby(electronics_data['event_time'].dt.month)['price'].sum()

# Create a line graph
plt.plot(jewelry_revenue_by_month.index, jewelry_revenue_by_month.values, marker='o', label='Jewelry')
plt.plot(electronics_revenue_by_month.index, electronics_revenue_by_month.values, marker='o', label='Electronics')

# Set the title and axis labels
plt.title('Revenue of Jewelry vs Electronics in 2020 by Month')
plt.xlabel('Month')
plt.ylabel('Revenue')

# Show the grid
plt.grid(True)

# Show the legend
plt.legend()

# Show the plot
plt.show()
