import pandas as pd
import matplotlib.pyplot as plt

def plot_material_donut_chart(data):
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
    plt.show()

# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')

# Call the function to plot the material donut chart
plot_material_donut_chart(jewelry_data)
