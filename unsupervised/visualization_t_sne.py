from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from unsupervised.clustering_training import cluster_analysis

def visualize_T_sne_clusters():
    df=cluster_analysis()
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(df.drop("Cluster", axis=1))

    # Plot t-SNE results
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df['Cluster'], palette='cool')
    plt.title("K-Means Clusters Visualized with t-SNE")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.savefig("data/t-SNE.png")