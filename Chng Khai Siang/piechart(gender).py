import pandas as pd
import matplotlib.pyplot as plt

def create_pie_chart(data):
    # Calculate the count of each gender
    gender_counts = data['gender'].value_counts()

    # Define the colours
    colours = ['red', 'blue']

    # Create a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(gender_counts, labels=gender_counts.index, colors=colours, autopct='%1.1f%%', startangle=90)
    plt.title('What gender purchase in jewelry store ')
    plt.axis('equal')
    plt.show()

def main():
    # Load the jewelry dataset
    jewelry_data = pd.read_csv('jewelry_dataset.csv')

    # Remove missing data
    jewelry_data = jewelry_data.dropna()

    # Call the create_pie_chart function
    create_pie_chart(jewelry_data)

if __name__ == '__main__':
    main()
