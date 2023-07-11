import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def recommend_graph():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('jewelry_dataset.csv')

    # Remove missing data
    jewelry_data = jewelry_data.dropna()
    
    # Group the data by category_code and colour, and calculate the count
    category_colour_counts = jewelry_data.groupby(['category_code', 'colour']).size().reset_index(name='count')

    # Pivot the data to have category_code as rows and colour as columns
    pivot_data = category_colour_counts.pivot(index='category_code', columns='colour', values='count')

    # Sort the categories based on the total count of colours in descending order
    sorted_categories = pivot_data.sum().sort_values(ascending=False).index

    # Reorder the columns based on the sorted categories
    pivot_data = pivot_data[sorted_categories]

    # Plot the colour distribution as a stacked bar chart
    plt.figure(figsize=(12, 8))
    sns.set_palette("husl")
    pivot_data.plot(kind='bar', stacked=True)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Colour Distribution by Category')
    plt.legend(title='Colour')
    plt.show()

# Call the function to create the recommended graph
recommend_graph()
