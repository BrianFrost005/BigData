import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def create_word_cloud(data):
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white')

    # Create a text string by joining all the unique words in the "gem" column
    unique_words = data['gem'].unique()
    text = ' '.join(unique_words)

    # Generate the word cloud for the combined text string
    wordcloud.generate(text)

    # Plot the word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud - Gem')
    plt.show()


# Load the jewelry dataset
jewelry_data = pd.read_csv('jewelry_dataset.csv')
jewelry_data = jewelry_data.dropna()

# Call the function to create the word cloud
create_word_cloud(jewelry_data)
