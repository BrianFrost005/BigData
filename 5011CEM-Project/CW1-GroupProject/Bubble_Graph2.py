# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 03:26:35 2023

@author: Tristan Tan
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the dataset
df = pd.read_csv('C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv')

# Filter the 'view' events
df_view = df[df['event_type'] == 'view']

# Group the data by brand and calculate the count of views and average price
grouped_brand = df_view.groupby('brand').agg({'event_type': 'count', 'price': 'mean'}).reset_index()
grouped_brand.rename(columns={'event_type': 'view_count', 'price': 'average_price'}, inplace=True)

# Sort the brands by view count in descending order
grouped_brand.sort_values('view_count', ascending=False, inplace=True)

# Select the top 10 brands
top_10_brands = grouped_brand.head(10)

# Define a color gradient for the brands
colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_10_brands)))

# Create a larger figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create a bubble graph with top 10 brands
for i, brand in enumerate(top_10_brands['brand']):
    ax.scatter(brand, top_10_brands['average_price'].iloc[i], s=top_10_brands['view_count'].iloc[i],
               c=[colors[i]], alpha=0.7)
    ax.text(brand, top_10_brands['average_price'].iloc[i], brand, ha='center', va='center')

ax.set_xlabel('Brand')
ax.set_ylabel('Average Price')
ax.set_title('Top 10 Brands Average Price by Popularity')

# Show the colorbar
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=len(top_10_brands)-1))
sm.set_array([])  # Empty array to avoid error
plt.colorbar(sm, label='Brand Index', ax=ax)

# Adjust the layout and spacing
plt.tight_layout()

# Show the plot
plt.show()