import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Filter the data for the year 2021
jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
jewelry_data = jewelry_data[jewelry_data['event_time'].dt.year == 2021]

# Filter out missing values for jewelry dataset
jewelry_data = jewelry_data.dropna(subset=['category_code'])

# Count the number of purchases in each jewelry category
jewelry_category_counts = jewelry_data['category_code'].value_counts()

# Select the top 10 jewelry categories with the highest counts
top_jewelry_categories = jewelry_category_counts.head(10)

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 6))

# Define a custom colormap for the jewelry categories
custom_colors = ['#FF3366', '#33FF99', '#66CCFF', '#FF9933', '#9966FF',
                 '#FFCC00', '#00FF66', '#0033FF', '#FF6600', '#6600FF']

# Create the bar chart for jewelry categories
jewelry_bars = ax.bar(top_jewelry_categories.index, top_jewelry_categories.values, color=custom_colors)

# Add labels to each jewelry bar
for bar in jewelry_bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

# Set the x-axis tick labels rotation
plt.xticks(rotation=45, ha='right')

# Set the title and axis labels
plt.title('Top 10 Jewelry Categories in 2021 vs Number of Purchases')
plt.xlabel('Category')
plt.ylabel('Number of Purchases')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
