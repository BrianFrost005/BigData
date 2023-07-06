# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 09:26:42 2023

@author: Tristan Tan
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Read the dataset
df = pd.read_csv('C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv')

# Filter the 'view', 'purchase', and 'cart' events
df_view = df[df['event_type'] == 'view']
df_purchase = df[df['event_type'] == 'purchase']
df_cart = df[df['event_type'] == 'cart']

# Group the data for each event type by product_id and calculate the number of events and average price
grouped_view = df_view.groupby('product_id').agg({'price': 'mean', 'event_type': 'count'}).reset_index()
grouped_view.rename(columns={'price': 'average_price', 'event_type': 'view_count'}, inplace=True)

grouped_purchase = df_purchase.groupby('product_id').agg({'price': 'mean', 'event_type': 'count'}).reset_index()
grouped_purchase.rename(columns={'price': 'average_price', 'event_type': 'purchase_count'}, inplace=True)

grouped_cart = df_cart.groupby('product_id').agg({'price': 'mean', 'event_type': 'count'}).reset_index()
grouped_cart.rename(columns={'price': 'average_price', 'event_type': 'cart_count'}, inplace=True)

# Create the bubble graph
scatter_view = plt.scatter(grouped_view['average_price'], grouped_view['view_count'], s=grouped_view['view_count'], alpha=0.5)
scatter_purchase = plt.scatter(grouped_purchase['average_price'], grouped_purchase['purchase_count'], s=grouped_purchase['purchase_count'], alpha=0.5)
scatter_cart = plt.scatter(grouped_cart['average_price'], grouped_cart['cart_count'], s=grouped_cart['cart_count'], alpha=0.5)

# Set labels and title
plt.xlabel('Average Price')
plt.ylabel('Number of Events')
plt.title('Product Events by Price')

# Create custom legend handles with circular markers
legend_handles = [
    Line2D([], [], linestyle='None', marker='o', markersize=8, alpha=0.5, color=scatter_view.get_facecolor()[0]),
    Line2D([], [], linestyle='None', marker='o', markersize=8, alpha=0.5, color=scatter_purchase.get_facecolor()[0]),
    Line2D([], [], linestyle='None', marker='o', markersize=8, alpha=0.5, color=scatter_cart.get_facecolor()[0])
]

# Create custom legend labels
legend_labels = ['View', 'Purchase', 'Cart']

# Create the legend
plt.legend(legend_handles, legend_labels)

# Show the plot
plt.show()