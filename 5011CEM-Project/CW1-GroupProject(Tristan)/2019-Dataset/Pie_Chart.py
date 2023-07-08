# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:37:41 2023

@author: Tristan Tan
"""

import pandas as pd
import matplotlib.pyplot as plt

# Assuming your dataset is stored in a pandas DataFrame called 'data'
# Load the dataset
data = pd.read_csv('C:\\Users\\Tristan Tan\\.spyder-py3\\5011CEM\\2019-Oct.csv\\2019-Oct.csv')

fig = plt.figure(figsize=(15, 17))

# Count the occurrences of each brand
brand_counts = data['brand'].value_counts()

# Select the top 10 brands
top_10_brands = brand_counts.head(10)

ax1 = fig.add_subplot(3,1,1)
ax1.pie(top_10_brands, labels=top_10_brands.index, autopct = '%1.1f%%')
ax1.set_title('Top 10 Brands')

# Count the occurrences of each brand
category_counts = data['category_code'].value_counts()

# Select the top 10 brands
top_10_category = category_counts.head(10)

ax2 = fig.add_subplot(3,1,2)
ax2.pie(top_10_category, labels=top_10_category.index, autopct = '%1.1f%%')
ax2.set_title('Top 10 Categories')

event_type = data['event_type'].value_counts()

ax3 = fig.add_subplot(3,1,3)
ax3.pie(event_type, labels = event_type.index, autopct = '%1.1f%%')
ax3.set_title('Event Type Pie Chart')

# Plotting the pie chart
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()
