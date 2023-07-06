import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the electronic dataset
electronic_data = pd.read_csv('electronic_dataset.csv')

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Concatenate the datasets
combined_data = pd.concat([electronic_data, jewelry_data])

# Select the columns for the heatmap
columns = ['category_code', 'brand', 'price']

# Filter the combined data for the selected columns
heatmap_data = combined_data[columns]

# Preprocess the data by label encoding categorical variables
label_encoder = LabelEncoder()
for column in ['category_code', 'brand']:
    heatmap_data[column] = label_encoder.fit_transform(heatmap_data[column])

# Calculate the correlation matrix
correlation_matrix = heatmap_data.corr()

# Create the heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Variables Correlation')
plt.show()
