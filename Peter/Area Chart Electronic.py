import pandas as pd
import matplotlib.pyplot as plt

# Load a subset of the electronic dataset for faster processing
df_electronics = pd.read_csv('modified_electronic.csv', nrows=10000)

# Data preprocessing
# Convert event_time to datetime
df_electronics['event_time'] = pd.to_datetime(df_electronics['event_time'])

# Extract year from event_time
df_electronics['year'] = df_electronics['event_time'].dt.year

# Filter the data for a specific year (e.g., 2020)
year = 2020
df_filtered = df_electronics[df_electronics['year'] == year]

# Handling missing values
df_filtered.dropna(subset=['category_code'], inplace=True)

# Get the top electronic categories by number of purchases
top_categories = df_filtered['category_code'].value_counts().head(10)

# Create the area graph
plt.fill_between(top_categories.index, top_categories.values, alpha=0.5)

# Add labels and title to the graph
plt.xlabel('Category'.upper(), fontweight='bold', fontsize=14)
plt.ylabel('Number of Purchases'.upper(), fontweight='bold', fontsize=12)
plt.title('Top Electronic Categories and Purchases in {}'.format(year).upper(), fontweight='bold', fontsize=14)

# Rotate the x-axis tick labels for better readability
plt.xticks(rotation=45)

# Display the area graph
plt.show()


