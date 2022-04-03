from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


def get_cluster_data(data):
    model = AgglomerativeClustering(
        n_clusters=2,
        linkage='complete',
        affinity='manhattan'
    )

    model.fit(data)
    labels = list(model.labels_)
    return labels


def draw_figure(data):
    plt.figure(figsize=(10, 7))
    plt.title("Tweets Dendograms")
    shc.dendrogram(shc.linkage(data, method='ward'))
    plt.axhline(y=46, color='r', linestyle='--')
    plt.savefig('tweets_dendograms')
    plt.show()


if __name__ == '__main__':
    import data_model
    # print(get_cluster_data(data_model.get_frequency_data()))
    draw_figure(data_model.get_frequency_data())

