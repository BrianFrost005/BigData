import pandas as pd
import matplotlib.pyplot as plt

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Convert event_time to datetime
jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])

# Extract the year from event_time
jewelry_data['year'] = jewelry_data['event_time'].dt.year

# Filter out missing values and negative prices
jewelry_data = jewelry_data.dropna(subset=['price'])
jewelry_data = jewelry_data[jewelry_data['price'] > 0]

# Calculate the revenue of jewelry for each year
revenue_by_year = jewelry_data.groupby('year')['price'].sum()

# Create a line graph
plt.plot(revenue_by_year.index, revenue_by_year.values, marker='o')

# Set the title and axis labels
plt.title('Revenue of Jewelry by Year')
plt.xlabel('Year')
plt.ylabel('Revenue')

# Show the grid
plt.grid(True)

# Show the plot
plt.show()
