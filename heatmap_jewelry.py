import pandas as pd
import seaborn as sns

#load data
dff = pd.read_csv('modified2.csv')
#filter data
dff = dff[['product_id','order_id','price','user_id','brand_id']]

#set colors
colors = ['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']
custom_cmap = sns.color_palette(colors)

#calculate correlation matrix
corr2 = dff.corr()

#show heatmap
sns.heatmap(corr2, cmap=custom_cmap, vmin=0.01, vmax=0.1, annot=True)