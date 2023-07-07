# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:21:54 2023

@author: Tristan Tan
"""
'''
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv')

# Calculate the number of unique users at each stage of the funnel
num_visits = len(df['user_id'].unique())
num_views = len(df[df['event_type'] == 'view']['user_id'].unique())
num_add_to_cart = len(df[df['event_type'] == 'cart']['user_id'].unique())
num_purchase = len(df[df['event_type'] == 'purchase']['user_id'].unique())

# Define the funnel stages and their corresponding values
funnel_stages = ['Visits', 'Views', 'Add to Cart', 'Purchase']
funnel_values = [num_visits, num_views, num_add_to_cart, num_purchase]

# Calculate the conversion rates between stages
conversion_rates = [funnel_values[i] / funnel_values[i-1] for i in range(1, len(funnel_values))]

# Create the funnel chart
fig, ax = plt.subplots()
ax.barh(range(len(funnel_stages)), funnel_values, color='skyblue')

# Add labels to the funnel stages
for i, stage in enumerate(funnel_stages):
    ax.text(0.02, i, stage, ha='center', va='center', color='black')

# Add labels to the funnel values
for i, value in enumerate(funnel_values):
    ax.text(value + 10, i, str(value), ha='left', va='center', color='black')

# Add conversion rate labels
for i, rate in enumerate(conversion_rates):
    ax.text(funnel_values[i] / 2, i + 0.5, f'{rate:.2%}', ha='center', va='center', color='white')

# Set the axis labels and title
ax.set_xlabel('Number of Users')
ax.set_ylabel('Funnel Stage')
ax.set_title('Conversion Funnel')

# Invert the y-axis to display the stages from top to bottom
ax.invert_yaxis()

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the funnel chart
plt.show()'''
'''
#df = pd.read_csv('C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv')
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
dataset_path = 'C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv'
df = pd.read_csv(dataset_path)

# Filter out rows with missing values in the desired columns
filtered_df = df.dropna(subset=['category_code', 'brand', 'price'])

# Group by category, brand, and calculate the total price
grouped_df = filtered_df.groupby(['category_code', 'brand']).sum('price')

# Reset the index
grouped_df = grouped_df.reset_index()

# Sort the values by price in descending order
sorted_df = grouped_df.sort_values('price', ascending=False)

# Set up the donut chart
fig, ax = plt.subplots()

# Calculate the total price
total_price = sorted_df['price'].sum()

# Calculate the percentage for each category, brand
sorted_df['percentage'] = sorted_df['price'] / total_price * 100

# Create a list of colors for the wedges
colors = plt.cm.tab20c(range(len(sorted_df)))

# Plot the donut chart
ax.pie(sorted_df['percentage'], labels=sorted_df['category_code'] + ' - ' + sorted_df['brand'],
       startangle=90, counterclock=False, colors=colors, wedgeprops={'width': 0.3})

# Add a circle at the center to create a donut effect
centre_circle = plt.Circle((0, 0), 0.7, color='white')
ax.add_artist(centre_circle)

# Add a title
ax.set_title('Category-Brand Distribution (Price)')

# Set aspect ratio to be equal
ax.axis('equal')

# Display the chart
plt.show()'''

import pandas as pd
import matplotlib.pyplot as plt
import squarify

# Load the dataset
dataset_path = 'C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv'
df = pd.read_csv(dataset_path)

# Filter out rows with missing values in the desired columns
filtered_df = df.dropna(subset=['category_code', 'brand', 'price'])

# Group by category, brand, and calculate the total price
grouped_df = filtered_df.groupby(['category_code', 'brand']).sum('price')

# Reset the index
grouped_df = grouped_df.reset_index()

# Sort the values by price in descending order
sorted_df = grouped_df.sort_values('price', ascending=False)

# Select the top 10 brands based on price
top_10_brands = sorted_df.head(10)

# Prepare the data for treemap
labels = top_10_brands['category_code'] + ' - ' + top_10_brands['brand']
sizes = top_10_brands['price']

# Increase the canvas size of the treemap
fig = plt.figure(figsize=(19, 15))
ax = fig.add_subplot()
squarify.plot(sizes=sizes, label=labels, ax=ax)

# Add a title
ax.set_title('Top 10 Brands (Price) - Treemap')

# Remove axis labels
ax.set_xticks([])
ax.set_yticks([])

# Display the chart
plt.show()