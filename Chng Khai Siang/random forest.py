import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Drop rows with missing values
jewelry_data.dropna(inplace=True)

# Select features and target variable
features = jewelry_data[['product_id', 'brand_id', 'price', 'user_id']]
target = jewelry_data['category_code']

# Perform one-hot encoding on the "brand" column
features_encoded = pd.get_dummies(features, columns=['brand_id'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

# Choose a model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100)

# Model Training
model.fit(X_train, y_train)

# Feature Importance
importance = model.feature_importances_

# Sort feature importance in descending order
sorted_indices = importance.argsort()[::-1]
sorted_importance = importance[sorted_indices]

# Get the feature names
feature_names = features_encoded.columns[sorted_indices]

# Plot the feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, sorted_importance)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()
