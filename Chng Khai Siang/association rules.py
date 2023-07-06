import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import random
from textwrap import wrap

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Drop rows with missing values
jewelry_data.dropna(inplace=True)

# Select relevant columns
data = jewelry_data[['order_id', 'product_id', 'category_code']]

# Convert the data to a transactional format
transactions = data.groupby(['order_id', 'product_id'])['category_code'].apply(list).reset_index()

# Convert the transactional data to a boolean DataFrame
te = TransactionEncoder()
data_encoded = te.fit_transform(transactions['category_code'])
transactions_encoded = pd.DataFrame(data_encoded, columns=te.columns_)

# Generate frequent itemsets using Apriori algorithm
frequent_itemsets = apriori(transactions_encoded, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)

# Shuffle the association rules randomly
random_rules = rules.sample(frac=1).reset_index(drop=True)

# Select 20 random association rules
random_sample = random_rules.head(20)

# Plot the random sample of 20 rules by lift with wrapped text labels and light pink color for all bars
plt.figure(figsize=(10, 8))
antecedents_labels = ['\n'.join(wrap(', '.join(map(str, antecedent)), width=20)) for antecedent in random_sample['antecedents']]
plt.barh(antecedents_labels, random_sample['lift'], color='#FFC0CB')  # Light pink color code: #FFC0CB
plt.xlabel('Lift')
plt.ylabel('Antecedents')
plt.title('Random Sample of 20 Association Rules by Lift')
plt.tight_layout()
plt.show()
