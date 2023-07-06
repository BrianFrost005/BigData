import pandas as pd
import matplotlib.pyplot as plt

# Load the electronics dataset
electronics_data = pd.read_csv('electronic_dataset.csv')

# Convert event_time to datetime
electronics_data['event_time'] = pd.to_datetime(electronics_data['event_time'])

# Extract the year and month from event_time
electronics_data['year'] = electronics_data['event_time'].dt.year
electronics_data['month'] = electronics_data['event_time'].dt.month

# Filter the data for the year 2020
electronics_data = electronics_data[electronics_data['year'] == 2020]

# Filter out missing values and negative prices
electronics_data = electronics_data.dropna(subset=['price'])
electronics_data = electronics_data[electronics_data['price'] > 0]

# Calculate the revenue of electronics for each month in 2020
revenue_by_month = electronics_data.groupby('month')['price'].sum()

# Create a line graph
revenue_by_month.plot(marker='o')

# Set the title and axis labels
plt.title('Revenue of Electronics by Month in 2020')
plt.xlabel('Month')
plt.ylabel('Revenue')

# Show the grid
plt.grid(True)

# Show the plot
plt.show()
