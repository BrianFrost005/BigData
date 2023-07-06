import pandas as pd
import matplotlib.pyplot as plt

# Load electronic dataset
electronic_data = pd.read_csv("electronic_dataset.csv")
# Load jewelry dataset
jewelry_data = pd.read_csv("jewelry_dataset.csv")

#MANIPULATING THE DATA
#Check for missing data
print(jewelry_data.isna().sum())
print(electronic_data.isna().sum())

#Remove missing data
jewelry_data = jewelry_data.dropna()
electronic_data = electronic_data.dropna()

#save the new non-missing values file
jewelry_data.to_csv('modified_jewelry.csv')
electronic_data.to_csv('modified_electronics.csv')

#change all same values at once (main objective is to make names shorter)
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.tablet', 'tablet')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.audio.headphone', 'headphone')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.smartphone', 'smartphone')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.video.tv', 'tv')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.clocks', 'clocks')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.telephone', 'telephone')
electronic_data['category_code'] = electronic_data['category_code'].replace('electronics.tablet', 'tablet')

electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.personal.scales', 'scales')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.kitchen.refrigerators', 'refrigerators')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.kitchen.kettle', 'kettle')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.kitchen.blender', 'blender')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.kitchen.mixer', 'mixer')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.kitchen.washer', 'washer')
electronic_data['category_code'] = electronic_data['category_code'].replace('appliances.iron', 'iron')

#graph generation
#compare only certain categories in histogram 
#histogram compare sales of tablets, headphones, smartphones, tvs and clocks)
specific_categories = ['tablet', 'headphone', 'smartphone', 'tv', 'clocks']
filtered_electronic_data= electronic_data[electronic_data['category_code'].isin(specific_categories)]

# Create scatter plot for electronic dataset
plt.scatter(filtered_electronic_data['price'], filtered_electronic_data['category_code'], color='blue', label='Electronic')

# Create scatter plot for jewelry dataset
plt.scatter(jewelry_data['price'], jewelry_data['category_code'], color='red', label='Jewelry')

# Set plot title and labels
plt.title('Scatter Plot - Price vs Category')
plt.xlabel('Price')
plt.ylabel('Category')

# Add legend
plt.legend()

# Show the scatter plot
plt.show()
