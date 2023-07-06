import pandas as pd
import seaborn as sns

df = pd.read_csv('modified.csv')
df = df[['category_id','product_id','order_id','price','user_id','brand']]

colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
custom_cmap = sns.color_palette(colors)

corr = df.corr()
sns.heatmap(corr, cmap=custom_cmap, vmin=0, vmax=0.1, annot=True)



