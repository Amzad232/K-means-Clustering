# Clustering documents with Python

# 1. Fetch Wikipedia articles

# Using the wikipedia package it is very easy to download content from Wikipedia. For this example, we will use the content of the articles for:
# Data Science, Artificial intelligence,Machine Learning,European Central Bank
# BankFinancial, technology,International Monetary Fund,Basketball Swimming , Tennis

# The content of each Wikipedia article is stored wiki_list while the title of each article is stored in variable title.

import pandas as pd
#pip install wikipedia
import wikipedia

articles=['Data Science','Artificial intelligence','European Central Bank','Bank','Financial technology','International Monetary Fund','Basketball','Swimming']

wiki_lst = []
title = []

for article in articles:
    print("loading content:",article)
    wiki_lst.append(wikipedia.page(article).content)
    title.append(article)

print("examine content")

# 2. Represent each article as a vector

# Since we are going to use k-means, we need to represent each article as a numeric vector. A popular method is to use term-frequency/inverse-document-frequency (tf-idf). Put it simply, with this method for each word w and document d we calculate:
# tf(w,d): the ratio of the number of appearances of w in d divided with the total number of words in d.
# idf(w): the logarithm of the fraction of the total number of documents divided by the number of documents that contain w.
# tfidf(w,d)=tf(w,d) x idf(w)
# It is recommended that common, stop words are excluded. All the calculations are easily done with sklearn’s TfidfVectorizer.

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words={'english'})
X = vectorizer.fit_transform(wiki_lst)

# 3. Perform k-means clustering

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

sum_of_sqaured_distance = []
K = range(2,9)

for k in K:
    km = KMeans(n_clusters=k,max_iter=200,n_init=10)
    km = km.fit(X)
    sum_of_sqaured_distance.append(km.inertia_)

plt.plot(K,sum_of_sqaured_distance) # without -bx
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

plt.plot(K,sum_of_sqaured_distance, '-bx') # with -bx
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# The plot is almost a straight line, probably because we have to few articles. 
# But at a closer examination a dent appears for k=4 or k=5. 
# We will try to cluster into 5 groups.

new_k = 5
model = KMeans(n_clusters=new_k, init='k-means++', max_iter=200, n_init=10)
model.fit(X)
labels = model.labels_
wiki_cluster = pd.DataFrame(list(zip(title,labels)),columns=['title','cluster'])
print(wiki_cluster.sort_values(by=['cluster']))

#  Evaluate the result

# Since we have used only 10 articles, it is fairly easy to evaluate the clustering 
# just by examining what articles are contained in each cluster. 
# That would be difficult for a large corpus. A nice way is to create 
# a word cloud from the articles of each cluster.

# pip install wordcloud
from wordcloud import WordCloud

result={'cluster':labels,'wiki':wiki_lst}
result=pd.DataFrame(result)
for k in range(0,new_k):
   s=result[result.cluster==k]
   text=s['wiki'].str.cat(sep=' ')
   text=text.lower()
   text=' '.join([word for word in text.split()])
   wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
   print('Cluster: {}'.format(k))
   print('Titles')
   titles=wiki_cluster[wiki_cluster.cluster==k]['title']         
   print(titles.to_string(index=False))
   plt.figure()
   plt.imshow(wordcloud, interpolation="bilinear")
   plt.axis("off")
   plt.show()

















