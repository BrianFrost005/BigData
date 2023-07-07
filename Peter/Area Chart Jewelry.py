import pandas as pd
import matplotlib.pyplot as plt

# Load the jewelry dataset
df_jewelry = pd.read_csv('modified_jewelry.csv')

# Preprocessing for jewelry dataset
df_jewelry['event_time'] = pd.to_datetime(df_jewelry['event_time'])
df_jewelry['year'] = df_jewelry['event_time'].dt.year

# Get the top jewelry categories by number of purchases for each year
top_categories_by_year = df_jewelry.groupby(['year', 'category_code'])['category_code'].count().reset_index(name='Number of Purchases').sort_values(by=['year', 'Number of Purchases'], ascending=[True, False])

# Get the top categories for each year
top_categories_per_year = top_categories_by_year.groupby('year').head(10)

# Create a pivot table for easy plotting
pivot_table = top_categories_per_year.pivot(index='year', columns='category_code', values='Number of Purchases')

# Create the area graph
pivot_table.plot.area(alpha=0.8)

# Add labels and title to the graph
plt.xlabel('Year')
plt.ylabel('Number of Purchases')
plt.title('Top Jewelry Categories and Purchases by Year', fontweight='bold')

# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45)

# Display the area graph
plt.show()
