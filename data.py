#Look into clustering score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

data = pd.read_csv("data.csv")

text = data["PNT_ATRISKNOTES_TX"]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(text)

k = 50 #Number of clusters
kmeans = KMeans(n_clusters=k, random_state=41, n_init="auto")
kmeans.fit(X)
labels = kmeans.labels_

terms = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
top_words = [[terms[j] for j in order_centroids[i, :10]] for i in range(k)]

for i in range(k):
    print(f"Cluster {i} top terms: {top_words[i]}, length: {labels.tolist().count(i)}")


df = pd.DataFrame({'Text': text, 'Cluster': labels})
df_sorted = df.sort_values(by='Cluster')
df_sorted.to_csv("clustered.csv", index=False)