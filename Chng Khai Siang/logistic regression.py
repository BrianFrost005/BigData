import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
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

# Scale the features using StandardScaler
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# Logistic Regression with increased max_iter
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_y_pred = logistic_model.predict(X_test)
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
print('Logistic Regression Accuracy:', logistic_accuracy)

# Create confusion matrix
cm = confusion_matrix(y_test, logistic_y_pred)

# Plot confusion matrix as heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
