# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:56:16 2023

@author: Tristan Tan
"""

import pandas as pd
import matplotlib.pyplot as plt
import squarify
import matplotlib.cm as cm
import warnings

warnings.filterwarnings("ignore",category=UserWarning)

# Load the dataset
dataset_path = 'C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Nov.csv\\2019-Nov.csv'
df = pd.read_csv(dataset_path, nrows=1000000)

# Filter out rows with missing values in the desired columns
filtered_df = df.dropna(subset=['category_code', 'brand', 'price'])

# Group by category and brand, and count the occurrences
occurrence_df = filtered_df.groupby(['category_code', 'brand']).size().reset_index(name='occurrences')

# Sort by occurrences in descending order and select the top 10 category-brands
top_10_occurrences = occurrence_df.sort_values('occurrences', ascending=False).head(10)

# Filter the dataframe to include only the top 10 category-brands
filtered_top_10 = filtered_df.merge(top_10_occurrences, on=['category_code', 'brand'])

# Calculate the average price for each of the top 10 category-brands
average_prices = filtered_top_10.groupby(['category_code', 'brand'])['price'].mean().reset_index()

# Prepare the data for treemap
labels = average_prices['category_code'] + ' - ' + average_prices['brand']
sizes = average_prices['price']

# Set up color map and normalize the sizes
cmap = cm.get_cmap('YlOrRd')
norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
colors = [cmap(norm(value)) for value in sizes]

# Increase the canvas size of the treemap
fig = plt.figure(figsize=(19, 15))
ax = fig.add_subplot()

# Plot the treemap with colors
squarify.plot(sizes=sizes, label=labels, ax=ax, color=colors)

# Add a title
ax.set_title('Top 10 Category-Brands (Avg. Price) - Treemap')

# Remove axis labels
ax.set_xticks([])
ax.set_yticks([])

# Create a colorbar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
cbar.set_label('Price')

# Display the chart
plt.show()
