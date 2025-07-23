from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from unsupervised.clustering_training import cluster_analysis

def visualize_dimension_clusters():
    df=cluster_analysis()
    print("Cluster analysis completed. Visualizing clusters...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df.drop("Cluster", axis=1))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['Cluster'], palette='viridis')
    plt.title("K-Means Clusters Visualized with PCA")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    print("Displaying the plot...")
    plt.grid(True)
    plt.savefig("data/dimension-pca.png")
