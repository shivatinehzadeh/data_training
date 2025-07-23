from sklearn.cluster import KMeans
from data_load_and_cleaning import data_cleaning

def cluster_analysis():
    df=data_cleaning()
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(df)
    print(df['Cluster'].value_counts())
    return df