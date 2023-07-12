#conda activate streamlit-env
#D:\anaconda\big data\report
#streamlit run main.py

import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import networkx as nx
import warnings
import squarify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from textwrap import wrap
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from wordcloud import WordCloud

# Load the electronic dataset
electronic_data = pd.read_csv('modified.csv')

# Load the jewelry dataset
jewelry_data = pd.read_csv('modified2.csv')

# Function to create the first scatter plot
def scatter_plot1():
    # Change all same values at once (main objective is to make names shorter)
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
    
    # Filter electronic data for specific categories
    specific_categories = ['tablet', 'headphone', 'smartphone', 'tv', 'clocks']
    filtered_electronic_data = electronic_data[electronic_data['category_code'].isin(specific_categories)]
    
    # Create scatter plot for electronic dataset
    fig, ax = plt.subplots()
    ax.scatter(filtered_electronic_data['price'], filtered_electronic_data['category_code'], color='blue', label='Electronic')
    
    # Create scatter plot for jewelry dataset
    ax.scatter(jewelry_data['price'], jewelry_data['category_code'], color='red', label='Jewelry')
    
    # Set plot title and labels
    ax.set_title('Scatter Plot - Price vs Category')
    ax.set_xlabel('Price')
    ax.set_ylabel('Category')
    
    # Add legend
    ax.legend()
    
    # Show the scatter plot in Streamlit
    st.pyplot(fig)

# Function to create the second scatter plot
def scatter_plot2():
    # MANIPULATING THE DATA
    # Check for missing data
    jewelry_data.dropna(inplace=True)
    electronic_data.dropna(inplace=True)

    # Preprocess the electronic data
    electronic_data['event_time'] = pd.to_datetime(electronic_data['event_time'])
    electronic_data['month'] = electronic_data['event_time'].dt.month
    electronic_data['year'] = electronic_data['event_time'].dt.year

    # Preprocess the jewelry data
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    jewelry_data['month'] = jewelry_data['event_time'].dt.month
    jewelry_data['year'] = jewelry_data['event_time'].dt.year

    # Group the electronic data by category_code and calculate the total number of purchases
    electronic_grouped_data = electronic_data.groupby('category_code').size().reset_index(name='purchase_count_electronic')

    # Group the jewelry data by category_code and calculate the total number of purchases
    jewelry_grouped_data = jewelry_data.groupby('category_code').size().reset_index(name='purchase_count_jewelry')

    # Select the top five categories from each dataset
    top_categories_electronic = electronic_grouped_data.nlargest(5, 'purchase_count_electronic')['category_code'].tolist()
    top_categories_jewelry = jewelry_grouped_data.nlargest(5, 'purchase_count_jewelry')['category_code'].tolist()

    # Filter the electronic dataset to include only the top categories
    filtered_electronic_data = electronic_data[electronic_data['category_code'].isin(top_categories_electronic)]

    # Filter the jewelry dataset to include only the top categories
    filtered_jewelry_data = jewelry_data[jewelry_data['category_code'].isin(top_categories_jewelry)]

    # Concatenate the filtered datasets
    combined_data = pd.concat([filtered_electronic_data, filtered_jewelry_data])

    # Group the combined data by month, year, and category_code, and calculate the total number of purchases
    grouped_data = combined_data.groupby(['year', 'month', 'category_code']).size().reset_index(name='purchase_count')

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the scatter plot
    for category in top_categories_electronic:
        category_data = grouped_data[grouped_data['category_code'] == category]
        ax.scatter(category_data['purchase_count'], category_data['category_code'], label=category)

    for category in top_categories_jewelry:
        category_data = grouped_data[grouped_data['category_code'] == category]
        ax.scatter(category_data['purchase_count'], category_data['category_code'], label=category)

    ax.set_xlabel('Purchase Count')
    ax.set_ylabel('Category Code')
    ax.set_title('Scatter Plot of Purchase Count vs Category Code')
    ax.legend()
    ax.grid(True)

    # Show the scatter plot
    st.pyplot(fig)

# Function to create the third scatter plot
def scatter_plot3():
    # MANIPULATING THE DATA
    # Check for missing data
    jewelry_data.dropna(inplace=True)
    electronic_data.dropna(inplace=True)

    # Preprocess the electronic data
    electronic_data['event_time'] = pd.to_datetime(electronic_data['event_time'])
    electronic_data['month'] = electronic_data['event_time'].dt.month
    electronic_data['year'] = electronic_data['event_time'].dt.year

    # Preprocess the jewelry data
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    jewelry_data['month'] = jewelry_data['event_time'].dt.month
    jewelry_data['year'] = jewelry_data['event_time'].dt.year
    
    # Combine the datasets
    combined_data = pd.concat([electronic_data, jewelry_data])

    # Group the data by month, year, and category_code, and calculate the total number of purchases and the average price
    grouped_data = combined_data.groupby(['year', 'month']).agg({'price': 'mean', 'order_id': 'count'}).reset_index()
    grouped_data.rename(columns={'order_id': 'purchase_count'}, inplace=True)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the scatter plot
    ax.scatter(grouped_data['price'], grouped_data['purchase_count'])
    ax.set_xlabel('Price')
    ax.set_ylabel('Purchase Count')
    ax.set_title('Scatter Plot of Price vs Purchase Count')
    ax.grid(True)
    
    # Show the scatter plot
    st.pyplot(fig)

def areachart1():
    # Load a subset of the electronic dataset for faster processing
    df_electronics = pd.read_csv('modified.csv', nrows=10000)

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

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the area graph
    ax.fill_between(top_categories.index, top_categories.values, alpha=0.5)

    # Add labels and title to the graph
    ax.set_xlabel('Category'.upper(), fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of Purchases'.upper(), fontweight='bold', fontsize=12)
    ax.set_title('Top Electronic Categories and Purchases in {}'.format(year).upper(), fontweight='bold', fontsize=14)

    # Rotate the x-axis tick labels for better readability
    plt.xticks(rotation=45)

    # Adjust the layout
    plt.tight_layout()

    # Show the area graph
    st.pyplot(fig)

def areachart2():
    # Load the jewelry dataset
    df_jewelry = pd.read_csv('modified2.csv')

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
    plt.xlabel('Year'.upper(), fontweight='bold', fontsize=14)
    plt.ylabel('Number of Purchases'.upper(), fontweight='bold', fontsize=12)
    plt.title('Top Jewelry Categories and Purchases by Year'.upper(), fontweight='bold', fontsize=14)

    # Rotate the x-axis tick labels for better readability
    plt.xticks(rotation=45)

    # Display the area graph
    st.pyplot(plt.gcf())  # Pass the figure object plt.gcf() to st.pyplot()

def areachart3():
    # Load the jewelry dataset
    df_jewelry = pd.read_csv('modified2.csv')

    # Preprocessing for jewelry dataset
    # Convert event_time to datetime
    df_jewelry['event_time'] = pd.to_datetime(df_jewelry['event_time'])
    # Extract year from event_time
    df_jewelry['year'] = df_jewelry['event_time'].dt.year

    # Load a subset of the electronic dataset for faster processing
    df_electronics = pd.read_csv('modified.csv', nrows=10000)

    # Preprocessing for electronics dataset
    # Convert event_time to datetime
    df_electronics['event_time'] = pd.to_datetime(df_electronics['event_time'])
    # Extract year from event_time
    df_electronics['year'] = df_electronics['event_time'].dt.year

    # Perform necessary calculations and aggregations
    jewelry_counts = df_jewelry.groupby(['year', 'category_code']).size().reset_index(name='Number of Purchases')
    electronics_counts = df_electronics.groupby(['year', 'category_code']).size().reset_index(name='Number of Purchases')

    # Select the top categories for each year
    jewelry_top_categories = jewelry_counts.groupby('year').apply(lambda x: x.nlargest(10, 'Number of Purchases')).reset_index(drop=True)
    electronics_top_categories = electronics_counts.groupby('year').apply(lambda x: x.nlargest(10, 'Number of Purchases')).reset_index(drop=True)

    # Create a pivot table for easy plotting
    jewelry_pivot = jewelry_top_categories.pivot(index='year', columns='category_code', values='Number of Purchases').fillna(0)
    electronics_pivot = electronics_top_categories.pivot(index='year', columns='category_code', values='Number of Purchases').fillna(0)

    # Create the area chart
    fig, ax = plt.subplots(figsize=(10, 6))
    jewelry_pivot.plot.area(ax=ax, alpha=0.9)
    electronics_pivot.plot.area(ax=ax, alpha=0.5)

    # Add labels and title to the graph
    st.xlabel('Year and month'.upper(), fontweight='bold', fontsize=14)
    st.ylabel('Number of Purchases'.upper(), fontweight='bold', fontsize=14)
    st.title('Comparison of Purchases: Jewelry vs. Electronics'.upper(), fontweight='bold', fontsize=14)

    # Rotate the x-axis tick labels for better readability
    plt.xticks(rotation=45)

    # Adjust the legend position
    plt.legend(loc='upper left')

    # Display the area chart
    plt.tight_layout()
    st.pyplot(fig)
    
#function to create bar chart1
def barchart1():
    # Filter the data for the year 2018
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    filtered_data = jewelry_data[jewelry_data['event_time'].dt.year == 2018]

    # Filter out missing values for jewelry dataset
    filtered_data = filtered_data.dropna(subset=['category_code'])

    # Count the number of purchases in each jewelry category
    jewelry_category_counts = filtered_data['category_code'].value_counts()

    # Select the top 10 jewelry categories with the highest counts
    top_jewelry_categories = jewelry_category_counts.head(10)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom colormap for the jewelry categories
    custom_colors = ['#FF3366', '#33FF99', '#66CCFF', '#FF9933', '#9966FF',
                     '#FFCC00', '#00FF66', '#0033FF', '#FF6600', '#6600FF']

    # Create the bar chart for jewelry categories
    jewelry_bars = ax.bar(top_jewelry_categories.index, top_jewelry_categories.values, color=custom_colors)

    # Add labels to each jewelry bar
    for bar in jewelry_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

    # Set the x-axis tick labels rotation
    plt.xticks(rotation=45, ha='right')

    # Set the title and axis labels
    plt.title('Top 10 Jewelry Categories in 2018 vs Number of Purchases')
    plt.xlabel('Category')
    plt.ylabel('Number of Purchases')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

#function to create bar chart2
def barchart2():
    # Filter the data for the year 2019
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    filtered_data = jewelry_data[jewelry_data['event_time'].dt.year == 2019]

    # Filter out missing values for jewelry dataset
    filtered_data = filtered_data.dropna(subset=['category_code'])

    # Count the number of purchases in each jewelry category
    jewelry_category_counts = filtered_data['category_code'].value_counts()

    # Select the top 10 jewelry categories with the highest counts
    top_jewelry_categories = jewelry_category_counts.head(10)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom colormap for the jewelry categories
    custom_colors = ['#FF3366', '#33FF99', '#66CCFF', '#FF9933', '#9966FF',
                     '#FFCC00', '#00FF66', '#0033FF', '#FF6600', '#6600FF']

    # Create the bar chart for jewelry categories
    jewelry_bars = ax.bar(top_jewelry_categories.index, top_jewelry_categories.values, color=custom_colors)

    # Add labels to each jewelry bar
    for bar in jewelry_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

    # Set the x-axis tick labels rotation
    plt.xticks(rotation=45, ha='right')

    # Set the title and axis labels
    plt.title('Top 10 Jewelry Categories in 2019 vs Number of Purchases')
    plt.xlabel('Category')
    plt.ylabel('Number of Purchases')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)
    
#function to create bar chart3
def barchart3():
    # Filter the data for the year 2020
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    filtered_data = jewelry_data[jewelry_data['event_time'].dt.year == 2020]

    # Filter out missing values for jewelry dataset
    filtered_data = filtered_data.dropna(subset=['category_code'])

    # Count the number of purchases in each jewelry category
    jewelry_category_counts = filtered_data['category_code'].value_counts()

    # Select the top 10 jewelry categories with the highest counts
    top_jewelry_categories = jewelry_category_counts.head(10)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom colormap for the jewelry categories
    custom_colors = ['#FF3366', '#33FF99', '#66CCFF', '#FF9933', '#9966FF',
                     '#FFCC00', '#00FF66', '#0033FF', '#FF6600', '#6600FF']

    # Create the bar chart for jewelry categories
    jewelry_bars = ax.bar(top_jewelry_categories.index, top_jewelry_categories.values, color=custom_colors)

    # Add labels to each jewelry bar
    for bar in jewelry_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

    # Set the x-axis tick labels rotation
    plt.xticks(rotation=45, ha='right')

    # Set the title and axis labels
    plt.title('Top 10 Jewelry Categories in 2020 vs Number of Purchases')
    plt.xlabel('Category')
    plt.ylabel('Number of Purchases')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

#function to create bar chart4
def barchart4():
    # Filter the data for the year 2021
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])
    filtered_data = jewelry_data[jewelry_data['event_time'].dt.year == 2021]

    # Filter out missing values for jewelry dataset
    filtered_data = filtered_data.dropna(subset=['category_code'])

    # Count the number of purchases in each jewelry category
    jewelry_category_counts = filtered_data['category_code'].value_counts()

    # Select the top 10 jewelry categories with the highest counts
    top_jewelry_categories = jewelry_category_counts.head(10)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a custom colormap for the jewelry categories
    custom_colors = ['#FF3366', '#33FF99', '#66CCFF', '#FF9933', '#9966FF',
                     '#FFCC00', '#00FF66', '#0033FF', '#FF6600', '#6600FF']

    # Create the bar chart for jewelry categories
    jewelry_bars = ax.bar(top_jewelry_categories.index, top_jewelry_categories.values, color=custom_colors)

    # Add labels to each jewelry bar
    for bar in jewelry_bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, height, ha='center', va='bottom')

    # Set the x-axis tick labels rotation
    plt.xticks(rotation=45, ha='right')

    # Set the title and axis labels
    plt.title('Top 10 Jewelry Categories in 2021 vs Number of Purchases')
    plt.xlabel('Category')
    plt.ylabel('Number of Purchases')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

def recommend_graph():  
    
    # Group the data by category_code and colour, and calculate the count
    category_colour_counts = jewelry_data.groupby(['category_code', 'colour']).size().reset_index(name='count')

    # Pivot the data to have category_code as rows and colour as columns
    pivot_data = category_colour_counts.pivot(index='category_code', columns='colour', values='count')

    # Sort the categories based on the total count of colours in descending order
    sorted_categories = pivot_data.sum().sort_values(ascending=False).index

    # Reorder the columns based on the sorted categories
    pivot_data = pivot_data[sorted_categories]

    # Plot the colour distribution as a stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_palette("husl")
    pivot_data.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Colour Distribution by Category')
    ax.legend(title='Colour')

    # Show the plot in Streamlit
    st.pyplot(fig)
    
#function to plot pie chart
def piechart():
    data = pd.read_csv('2019-Oct behaviour data.csv', nrows=100000)
    data = data.dropna()
    
    fig = plt.figure(figsize=(20, 20))

    # Count the occurrences of each brand
    brand_counts = data['brand'].value_counts()

    # Select the top 10 brands
    top_10_brands = brand_counts.head(10)

    ax1 = fig.add_subplot(3,1,1)
    ax1.pie(top_10_brands, labels=top_10_brands.index, autopct='%1.1f%%')
    ax1.set_title('Top 10 Brands')

    # Count the occurrences of each brand
    category_counts = data['category_code'].value_counts()

    # Select the top 10 brands
    top_10_category = category_counts.head(10)

    ax2 = fig.add_subplot(3,1,2)
    ax2.pie(top_10_category, labels=top_10_category.index, autopct='%1.1f%%')
    ax2.set_title('Top 10 Categories')

    #event_type data
    event_type = data['event_type'].value_counts()

    ax3 = fig.add_subplot(3,1,3)
    ax3.pie(event_type, labels=event_type.index, autopct='%1.1f%%')
    ax3.set_title('Event Type Pie Chart')

    # Plotting the pie chart
    # Adjust the spacing between subplots
    plt.subplots_adjust(hspace=0.3)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Set the limits to magnify the pie chart
    plt.axis([-1.5, 1.5, -1.5, 1.5])  # Adjust the limits based on your desired magnification level

    # Show the plot
    st.pyplot(fig)
    
def piechart_gender():
    data = pd.read_csv('modified2.csv')
    
    # Calculate the count of each gender
    gender_counts = data['gender'].value_counts()

    # Define the colours
    colours = ['red', 'blue']

    # Create a pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(gender_counts, labels=gender_counts.index, colors=colours, autopct='%1.1f%%', startangle=90)
    ax.set_title('What gender purchase in jewelry store')
    ax.axis('equal')

    # Show the plot in Streamlit
    st.pyplot(fig)

def donutchart():
    data = pd.read_csv('jewelry.csv')
    
    # Filter the data to include only rows with non-missing values in the material column
    filtered_data = data.dropna(subset=['material'])

    # Count the number of occurrences of each material category
    material_counts = filtered_data['material'].value_counts()

    # Combine all categories except "gold" into a single category "Other"
    other_count = material_counts[~(material_counts.index == 'gold')].sum()
    material_counts = pd.Series({'gold': material_counts['gold'], 'Other': other_count})

    # Create the donut chart
    fig, ax = plt.subplots()
    wedges, text, autotext = ax.pie(material_counts, labels=material_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
    plt.setp(autotext, size=10, weight='bold')  # Adjust the size and weight of the percentage labels
    ax.set_title('Material of jewelry purchase in jewelry store')

    # Add a circle in the center to create the donut shape
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    ax.add_artist(center_circle)

    plt.axis('equal')

    # Show the plot in Streamlit
    st.pyplot(fig)

def linegraph1():

    # Convert event_time to datetime
    electronic_data['event_time'] = pd.to_datetime(electronic_data['event_time'])

    # Extract the year and month from event_time
    electronic_data['year'] = electronic_data['event_time'].dt.year
    electronic_data['month'] = electronic_data['event_time'].dt.month

    # Filter the data for the year 2020
    electronics_data = electronic_data[electronic_data['year'] == 2020]

    # Filter out missing values and negative prices
    electronics_data = electronics_data.dropna(subset=['price'])
    electronics_data = electronics_data[electronics_data['price'] > 0]

    # Calculate the revenue of electronics for each month in 2020
    revenue_by_month = electronics_data.groupby('month')['price'].sum()

    # Create a line graph
    fig, ax = plt.subplots()
    revenue_by_month.plot(marker='o', ax=ax)

    # Set the title and axis labels
    ax.set_title('Revenue of Electronics by Month in 2020')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')

    # Show the grid
    ax.grid(True)

    # Show the plot
    st.pyplot(fig)
    
#function to create line graph 2
def linegraph2():
    # Convert event_time to datetime
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])

    # Extract the year from event_time
    jewelry_data['year'] = jewelry_data['event_time'].dt.year

    # Filter out missing values and negative prices
    filtered_data = jewelry_data.dropna(subset=['price'])
    filtered_data = jewelry_data[jewelry_data['price'] > 0]

    # Calculate the revenue of jewelry for each year
    revenue_by_year = filtered_data.groupby('year')['price'].sum()

    # Create a line graph
    fig, ax = plt.subplots()
    ax.plot(revenue_by_year.index.values, revenue_by_year.values, marker='o')

    # Set the title and axis labels
    ax.set_title('Revenue of Jewelry by Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Revenue')

    # Show the grid
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

#function to crete line graph 3
def linegraph3():
    jewelry_data = pd.read_csv('modified2.csv')
    electronic_data = pd.read_csv('modified.csv')
    
    # Convert event_time to datetime for jewelry dataset
    jewelry_data['event_time'] = pd.to_datetime(jewelry_data['event_time'])

    # Convert event_time to datetime for electronics dataset
    electronic_data['event_time'] = pd.to_datetime(electronic_data['event_time'])

    # Filter the data for the year 2020 for jewelry dataset
    filtered_data = jewelry_data[jewelry_data['event_time'].dt.year == 2020]

    # Filter the data for the year 2020 for electronics dataset
    filtered2_data = electronic_data[electronic_data['event_time'].dt.year == 2020]

    # Filter out missing values and negative prices for jewelry dataset
    filtered_data = filtered_data.dropna(subset=['price'])
    filtered_data = filtered_data[filtered_data['price'] > 0]

    # Filter out missing values and negative prices for electronics dataset
    filtered2_data = filtered2_data.dropna(subset=['price'])
    filtered2_data = filtered2_data[filtered2_data['price'] > 0]

    # Calculate the revenue of jewelry for each month in 2020
    jewelry_revenue_by_month = filtered_data.groupby(filtered_data['event_time'].dt.month)['price'].sum()

    # Calculate the revenue of electronics for each month in 2020
    electronics_revenue_by_month = filtered2_data.groupby(filtered2_data['event_time'].dt.month)['price'].sum()

    # Create a line graph
    fig, ax = plt.subplots()
    ax.plot(jewelry_revenue_by_month.index, jewelry_revenue_by_month.values, marker='o', label='Jewelry')
    ax.plot(electronics_revenue_by_month.index, electronics_revenue_by_month.values, marker='o', label='Electronics')

    # Set the title and axis labels
    ax.set_title('Revenue of Jewelry vs Electronics in 2020 by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Revenue')

    # Show the grid
    ax.grid(True)

    # Show the legend
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)


#function for histogram
def histogram1():
    # Load new filtered data sets
    df = pd.read_csv('modified.csv')

    # Change all same values at once (main objective is to make names shorter)
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

    # Graph generation
    # Compare only certain categories in histogram
    # Histogram compares sales of tablets, headphones, smartphones, TVs, and clocks
    specific_categories = ['tablet', 'headphone', 'smartphone', 'tv', 'clocks']
    filtered_df = df[df['category_code'].isin(specific_categories)]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create the histogram
    ax.hist(filtered_df['category_code'], bins=len(specific_categories))
    ax.set_xlabel('Category Code')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Specific Category Products')

    # Show the plot
    st.pyplot(fig)

#function histogram2
def histogram2():
    # Load new filtered data sets
    df = pd.read_csv('modified.csv')

    specific_brands = ['samsung', 'huawei', 'apple', 'lg', 'intel', 'logitech', 'hp']
    filtered_df = df[df['brand'].isin(specific_brands)]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create the histogram
    ax.hist(filtered_df['brand'], bins=len(specific_brands))
    ax.set_xlabel('Brand')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Specific Brands')

    # Show the plot
    st.pyplot(fig)
    
#function histogram3
def histogram3():
    # Load new filtered data sets
    df = pd.read_csv('modified2.csv')

    #change all same values at once (main objective is to make names shorter)
    df['category_code'] = df['category_code'].replace('jewelry.pendant', 'pendant')
    df['category_code'] = df['category_code'].replace('jewelry.earring', 'earring')
    df['category_code'] = df['category_code'].replace('jewelry.bracelet', 'bracelet')
    df['category_code'] = df['category_code'].replace('jewelry.ring', 'ring')
    df['category_code'] = df['category_code'].replace('jeweley.necklace', 'necklace')
    df['category_code'] = df['category_code'].replace('jeweley.brooch', 'brooch')
    
    specific_category = ['pendant', 'earring', 'bracelet', 'ring', 'necklace', 'brooch']
    filtered_df = df[df['category_code'].isin(specific_category)]
    
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Create the histogram
    ax.hist(filtered_df['category_code'])
    ax.set_xlabel('Category_code')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of category_code for jewelry')

    # Show the plot
    st.pyplot(fig)
    
#histogram 4
def histogram4():
    # Load new filtered data sets
    df = pd.read_csv('modified2.csv')
    
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Specify the price range and number of bins
    price_range = (0, 2000)  # Set the desired price range
    num_bins = 20  # Adjust the number of bins as per your preference

    # Create the histogram with the specified price range and number of bins
    ax.hist(df['price'], bins=num_bins, range=price_range)
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Price for Jewelry')

    # Show the plot
    st.pyplot(fig)

#histogram 5
def histogram5():
    # Load new filtered data sets
    df = pd.read_csv('modified.csv')
    
    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Specify the price range and number of bins
    price_range = (0, 2000)  # Set the desired price range
    num_bins = 20  # Adjust the number of bins as per your preference

    # Create the histogram with the specified price range and number of bins
    ax.hist(df['price'], bins=num_bins, range=price_range)
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Price for Electronics')

    # Show the plot
    st.pyplot(fig)
    
#function to create bubble graph
def bubblegraph():
    # Read the dataset
    df = pd.read_csv('2019-Oct behaviour data.csv', nrows=1000000)
    df = df.dropna()

    # Filter the 'view' events
    df_view = df[df['event_type'] == 'view']

    # Group the data by brand and calculate the count of views and average price
    grouped_brand = df_view.groupby('brand').agg({'event_type': 'count', 'price': 'mean'}).reset_index()
    grouped_brand.rename(columns={'event_type': 'view_count', 'price': 'average_price'}, inplace=True)

    # Sort the brands by view count in descending order
    grouped_brand.sort_values('view_count', ascending=False, inplace=True)

    # Select the top 10 brands
    top_10_brands = grouped_brand.head(10)

    # Define a color gradient for the brands
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(top_10_brands)))

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a bubble graph with top 10 brands
    for i, brand in enumerate(top_10_brands['brand']):
        ax.scatter(brand, top_10_brands['average_price'].iloc[i], s=top_10_brands['view_count'].iloc[i],
                   c=[colors[i]], alpha=0.7)
        ax.text(brand, top_10_brands['average_price'].iloc[i], brand, ha='center', va='center')

    ax.set_xlabel('Brand')
    ax.set_ylabel('Average Price')
    ax.set_title('Top 10 Brands Average Price by Popularity')

    # Show the colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=0, vmax=len(top_10_brands)-1))
    sm.set_array([])  # Empty array to avoid error
    plt.colorbar(sm, label='Brand Index', ax=ax)

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    st.pyplot(fig)

def wordcloud():
    # Load the data
    data = pd.read_csv('modified2.csv')

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Create a text string by joining all the unique words in the "gem" column
    unique_words = data['gem'].unique()
    text = ' '.join(unique_words)

    # Generate the word cloud for the combined text string
    wordcloud.generate(text)

    # Plot the word cloud
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud - Gem')

    # Display the word cloud in Streamlit
    st.pyplot(fig)

def treemap():
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load the dataset
    df = pd.read_csv('2019-Nov behaviour data.csv', nrows=1000000)

    # Filter out rows with missing values in the desired columns
    filtered_df = df.dropna(subset=['category_code', 'brand', 'price'])

    # Group by category and brand, and count the occurrences
    occurrence_df = filtered_df.groupby(['category_code', 'brand']).size().reset_index(name='occurrences')

    # Sort by occurrences in descending order and select the top 10 category-brands
    top_10_occurrences = occurrence_df.sort_values('occurrences', ascending=False).head(10)

    # Filter the dataframe to include only the top 10 category-brands
    filtered_top_10 = filtered_df.merge(top_10_occurrences, on=['category_code', 'brand'])

    # Calculate the average price for each of the top 10 category-brands
    average_prices = filtered_top_10.groupby(['category_code', 'brand'])['price'].mean().reset_index()

    # Prepare the data for treemap
    labels = average_prices['category_code'] + ' - ' + average_prices['brand']
    sizes = average_prices['price']

    # Set up color map and normalize the sizes
    cmap = cm.get_cmap('YlOrRd')
    norm = plt.Normalize(vmin=sizes.min(), vmax=sizes.max())
    colors = [cmap(norm(value)) for value in sizes]

    # Increase the canvas size of the treemap
    fig = plt.figure(figsize=(19, 15))
    ax = fig.add_subplot()

    # Plot the treemap with colors
    squarify.plot(sizes=sizes, label=labels, ax=ax, color=colors)

    # Add a title
    ax.set_title('Top 10 Category-Brands (Avg. Price) - Treemap')

    # Remove axis labels
    ax.set_xticks([])
    ax.set_yticks([])

    # Create a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
    cbar.set_label('Price')

    # Display the chart in Streamlit
    st.pyplot(fig)

#function to create heatmap1
def heatmap1():
    electronic_data = pd.read_csv('electronics.csv', nrows=10000)
    
    filtered_data = electronic_data[['category_id','product_id','order_id','price','user_id']]

    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
    custom_cmap = sns.color_palette(colors)

    corr = filtered_data.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap=custom_cmap, vmin=0, vmax=0.1, annot=True, ax=ax)
    
    # Show the plot
    st.pyplot(fig)

#function to create heatmap2
def heatmap2():
    filtered_data = jewelry_data[['product_id','order_id','price','user_id','brand_id']]

    # Set colors
    colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
    custom_cmap = sns.color_palette(colors)

    # Calculate correlation matrix
    corr2 = filtered_data.corr()

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Show heatmap
    sns.heatmap(corr2, cmap=custom_cmap, vmin=0.01, vmax=0.1, annot=True, ax=ax)

    # Show the plot
    st.pyplot(fig)

#machine models
#random forest classification algo
def random_forest():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('modified2.csv')
    
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
    
    # Model Prediction
    y_pred = model.predict(X_test)
    
    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Show the plot in Streamlit
    st.pyplot(fig)

#association algo 
def association():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('modified2.csv')

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
    rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=1.0)

    # Create a network graph of association rules
    G = nx.DiGraph()
    for i in range(len(rules)):
        antecedents = ', '.join(rules.iloc[i]['antecedents'])
        consequents = ', '.join(rules.iloc[i]['consequents'])
        G.add_edge(antecedents, consequents)

    # Plot the network graph with increased spacing
    fig, ax = plt.subplots(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1.5)  # Increase the k value for more spacing
    nx.draw_networkx(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray')
    ax.set_title('Association Rules Network Graph')
    ax.axis('off')
    plt.tight_layout()

    # Show the graph in Streamlit
    st.pyplot(fig)
    
def association2():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('modified2.csv')

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

    # Display the association rules
    st.write(rules)

def association3():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('modified2.csv')

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

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    antecedents_labels = ['\n'.join(wrap(', '.join(map(str, antecedent)), width=20)) for antecedent in random_sample['antecedents']]
    ax.barh(antecedents_labels, random_sample['lift'], color='#FFC0CB')  # Light pink color code: #FFC0CB
    ax.set_xlabel('Lift')
    ax.set_ylabel('Antecedents')
    ax.set_title('Random Sample of 20 Association Rules by Lift')
    plt.tight_layout()

    # Show the plot in Streamlit
    st.pyplot(fig)
    
#logistic regression algo
def ligistic_regression():
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
    st.write('Logistic Regression Accuracy:', logistic_accuracy)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, logistic_y_pred)
    
    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_title('Confusion Matrix - Logistic Regression')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Show the plot in Streamlit
    st.pyplot(fig)

def logistic_regression():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('modified2.csv')

    # Select features and target variable
    features = jewelry_data[['product_id', 'brand_id', 'price', 'user_id']]
    target = jewelry_data['category_code']

    # Perform one-hot encoding on the "brand" column
    features_encoded = pd.get_dummies(features, columns=['brand_id'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_encoded, target, test_size=0.2, random_state=42)

    # Choose a model (Logistic Regression)
    model = LogisticRegression()

    # Model Training
    model.fit(X_train, y_train)

    # Model Prediction
    y_pred = model.predict(X_test)

    # Model Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    st.write('Accuracy:', accuracy)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix as heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    # Show the plot in Streamlit
    st.pyplot(fig)
    
# Create the Streamlit app
def main():
    st.title('Graph Selector')

    # Create buttons for different scatter plots
    button_bar = st.button('Bar Chart')
    button_pie = st.button('Pie Chart')
    button_line = st.button('Line Graph')
    button_hist = st.button('Histogram')
    button_donut = st.button('Donut Chart')
    button_scatter = st.button('Scatter Plot')
    button_area = st.button('Area Chart')
    button_bubble = st.button('Bubble Graph')
    button_cloud = st.button('Word Cloud')
    button_tree = st.button('Tree Map')
    button_heat = st.button('Heatmap')
    button_rand = st.button('Random Forest Classification')
    button_assoc = st.button('Association')
    #button_logistic = st.button('Logistic Regression')
   
    # Handle button clicks
    if button_scatter:
        scatter_plot1()
        scatter_plot2()
        scatter_plot3()
    elif button_pie:
        piechart()
        piechart_gender()
    elif button_area:
        areachart1()
        areachart2()
        areachart3()
    elif button_bar:
        barchart1()
        barchart2()
        barchart3()
        barchart4()
        recommend_graph()
    elif button_line:
        linegraph1()
        linegraph2()
        #linegraph3()
    elif button_hist:
        histogram1()
        histogram2()
        histogram5()
        histogram3()
        histogram4()
    elif button_donut:
        donutchart()
    elif button_bubble:
        bubblegraph()
    elif button_cloud:
        wordcloud()
    elif button_heat:
        heatmap1()
        heatmap2()
    elif button_tree:
        treemap()
    elif button_rand:
        random_forest()
    elif button_assoc:
       #association()
        association2()
        association3()
   # elif button_logistic:
        #logistic_regression()

# Run the Streamlit app
if __name__ == '__main__':
    main()