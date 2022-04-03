from sklearn.mixture import GaussianMixture


def get_cluster_data(data):
    model = GaussianMixture(n_components=2)
    model.fit(data)
    labels = model.predict(data)

    return labels


if __name__ == '__main__':
    import data_model
    print(get_cluster_data(data_model.get_frequency_data()))

