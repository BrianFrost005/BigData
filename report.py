import pandas as pd
import matplotlib.pyplot as plt

# Load big data into a Pandas dataframe
#df = pd.read_excel('electronics (recovered).xlsx')
df = pd.read_csv('electronics.csv')
dff = pd.read_csv('jewelry.csv')


#LOOKING AT THE DATA
# Check the shape of the dataframe, no. of rows and columns
print(df.shape)
print(dff.shape)
# Check columns 
print(df.columns)
print(dff.columns)
# View the first 5 rows of the dataframe
print(df.head())
print(dff.head())


#MANIPULATING THE DATA
# Check for missing data
print(df.isna().sum())
print(dff.isna().sum())

# Remove missing data
df = df.dropna()
dff = dff.dropna()

# View descriptive statistics of the dataframe
print(df.describe())
print(dff.describe())

#drop columns that will not be utilized
#electronics
df.drop('product_id', axis=1, inplace=True)
df.drop('category_id', axis=1, inplace=True)
#jewelry
dff.drop('product_id', axis=1, inplace=True)
dff.drop('category_id', axis=1, inplace=True)
dff.drop('brand_id', axis=1, inplace=True)
dff.drop('material', axis=1, inplace=True)
dff.drop('gem', axis=1, inplace=True)

#save the new non-missing values file 
df.to_csv('modified.csv')
dff.to_csv('modified2.csv')

#load new filtered data sets
df = pd.read_csv('modified.csv')
dff = pd.read_csv('modified2.csv')

#convert date time format
df['event_time'] = pd.to_datetime(df['event_time'], format='%Y-%m-%d %H:%M:%S %Z')
dff['event_time'] = pd.to_datetime(dff['event_time'], format='%Y-%m-%d %H:%M:%S %Z')

#list out types of category
unique_categories = df['category_code'].unique()
print(unique_categories)
unique_categories = dff['category_code'].unique()
print(unique_categories)

#change all same values at once (main objective is to make names shorter)
df['category_code'] = df['category_code'].replace('electronics.tablet', 'tablet')
df['category_code'] = df['category_code'].replace('electronics.audio.headphone', 'headphone')
df['category_code'] = df['category_code'].replace('electronics.smartphone', 'smartphone')
df['category_code'] = df['category_code'].replace('electronics.video.tv', 'tv')
df['category_code'] = df['category_code'].replace('electronics.clocks', 'clocks')
df['category_code'] = df['category_code'].replace('electronics.telephone', 'telephone')
df['category_code'] = df['category_code'].replace('electronics.tablet', 'tablet')

df['category_code'] = df['category_code'].replace('appliances.personal.scales', 'scales')
df['category_code'] = df['category_code'].replace('appliances.kitchen.refrigerators', 'refrigerators')
df['category_code'] = df['category_code'].replace('appliances.kitchen.kettle', 'kettle')
df['category_code'] = df['category_code'].replace('appliances.kitchen.blender', 'blender')
df['category_code'] = df['category_code'].replace('appliances.kitchen.mixer', 'mixer')
df['category_code'] = df['category_code'].replace('appliances.kitchen.washer', 'washer')
df['category_code'] = df['category_code'].replace('appliances.iron', 'iron')

#graph generation
#compare only certain categories in histogram 
#histogram compare sales of tablets, headphones, smartphones, tvs and clocks)
specific_categories = ['tablet', 'headphone', 'smartphone', 'tv', 'clocks']
filtered_df = df[df['category_code'].isin(specific_categories)]
plt.hist(filtered_df['category_code'], bins=len(specific_categories))
plt.xlabel('category_code')
plt.ylabel('Frequency')
plt.title('Histogram of Specific category products')
plt.show()
#only the first x number of rows will be input into histogram
num_rows_to_display = 10000
display_df = filtered_df.head(num_rows_to_display)
plt.hist(display_df['category_code'])
plt.xlabel('category_code')
plt.ylabel('Frequency')
plt.title('Histogram of Specific category products in first 10000 rows')
plt.show()

#list out brands
unique_brands = df['brand'].unique()
print(unique_brands)

#compare only certain brands in histogram 
#histogram compare sales of samsung, huawei, apple, lg, intel, logitech, hp)
specific_brands = ['samsung', 'huawei', 'apple', 'lg', 'intel', 'logitech', 'hp']
filtered_df = df[df['brand'].isin(specific_brands)]
plt.hist(filtered_df['brand'], bins=len(specific_brands))
plt.xlabel('brand')
plt.ylabel('Frequency')
plt.title('Histogram of Specific brands')
plt.show()
#only the first x number of rows will be input into histogram
num_rows_to_display = 10000
display_df = filtered_df.head(num_rows_to_display)
plt.hist(display_df['brand'])
plt.xlabel('brand')
plt.ylabel('Frequency')
plt.title('Histogram of Specific brands in first 10000 rows')
plt.show()

#filter by date
#specific_date = pd.to_datetime('2020-04-24').date()
#filtered_df = df[df['event_time'].dt.date == specific_date]
#print(filtered_df.head())
#filter by time
#specific_time = pd.to_datetime('15:45:00').time()
#filtered_df = df[df['event_time'].dt.time == specific_time]
#print(filtered_df.head())
#filter by month
# Filter rows for a specific month
#specific_month = 2
#filtered_df = df[df['event_time'].dt.month == specific_month]
#print(filtered_df.head())
# Filter rows for a specific duration (from February to April)
#start_month = pd.Period('2022-02', freq='M')
#end_month = pd.Period('2022-04', freq='M')
#filtered_df = df[df['DateTime'].dt.to_period('M').between(start_month, end_month)]

#compare certain categories on a certain month span
start_month = pd.Period('2020-04', freq='M')
end_month = pd.Period('2020-04', freq='M')
specific_categories = ['tablet', 'headphone', 'smartphone', 'tv', 'clocks']
filtered_df = df[df['category_code'].isin(specific_categories) & df['event_time'].dt.to_period('M').between(start_month, end_month)]
plt.hist(filtered_df['category_code'])
plt.xlabel('category_code')
plt.ylabel('Frequency')
plt.title('Histogram of Specific category products')
plt.show()

#compare certain brand on a certain month span
start_month = pd.Period('2020-04', freq='M')
end_month = pd.Period('2020-07', freq='M')
specific_brands = ['samsung', 'huawei', 'apple', 'lg', 'intel', 'logitech', 'hp']
filtered_df = df[df['brand'].isin(specific_brands) & df['event_time'].dt.to_period('M').between(start_month, end_month)]
plt.hist(filtered_df['brand'])
plt.xlabel('brand')
plt.ylabel('Frequency')
plt.title('Histogram of Specific category products')
plt.show()

start_month = pd.Period('2020-08', freq='M')
end_month = pd.Period('2020-11', freq='M')
specific_brands = ['samsung', 'huawei', 'apple', 'lg', 'intel', 'logitech', 'hp']
filtered_df = df[df['brand'].isin(specific_brands) & df['event_time'].dt.to_period('M').between(start_month, end_month)]
plt.hist(filtered_df['brand'])
plt.xlabel('brand')
plt.ylabel('Frequency')
plt.title('Histogram of Specific category products')
plt.show()

# Filter rows for a specific day within a period of time
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-02-28')
specific_day = 5
filtered_df = df[(df['DateTime'] >= start_date) & (df['DateTime'] <= end_date) & (df['DateTime'].dt.day == specific_day)]