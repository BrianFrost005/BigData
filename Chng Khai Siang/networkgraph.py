import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the electronic dataset
electronic_data = pd.read_csv('electronic_dataset.csv')

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

#MANIPULATING THE DATA
#Check for missing data
print(jewelry_data.isna().sum())
print(electronic_data.isna().sum())

#Remove missing data
jewelry_data = jewelry_data.dropna()
electronic_data = electronic_data.dropna()

# Combine the datasets
combined_data = pd.concat([electronic_data, jewelry_data])

# Select the columns for the network graph
columns = ['user_id', 'category_code']

# Filter the combined data for the selected columns
network_data = combined_data[columns]

# Preprocess the data if needed

# Reduce the dataset size
network_data = network_data.sample(frac=0.1, random_state=42)

# Create a directed graph
G = nx.from_pandas_edgelist(network_data, source='user_id', target='category_code', create_using=nx.DiGraph)

# Plot the network graph using the random layout
plt.figure(figsize=(12, 8))
pos = nx.random_layout(G)
nx.draw_networkx(G, pos=pos, with_labels=True, node_size=3000, node_color='lightblue', edge_color='gray')
plt.title('Network Graph of User-Category Connections')
plt.axis('off')
plt.show()

