import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


def euclidean_distances(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2)**2))


def mahalanobis_distance(vec1, vec2):
    combine_vector = np.vstack([vec1, vec2])
    combine_vector = combine_vector.T
    return pdist(combine_vector, 'mahalanobis')


ALGORITHM_FUNC = {'euclidean': euclidean_distances,
                  'mahalanobis': mahalanobis_distance}


def two_by_two_distances(n_vec, algorithm_type, save_to_file='', save_to_figure=''):
    """
    Calculate the distance of every two vectors in an n-dimensional vector
    :param n_vec:
    :param algorithm_type:
    :param save_to_file:
    :param save_to_figure:
    :return: [[row_index, column_index, distance], ...]
    """
    if algorithm_type not in ALGORITHM_FUNC:
        return
    vec_count = len(n_vec)
    results = []
    i = 0
    j = 0
    for vec1, vec2 in list(itertools.combinations(n_vec, 2)):
        dist = ALGORITHM_FUNC[algorithm_type](vec1, vec2)
        if j != vec_count-1:
            j += 1
        else:
            i += 1
            j = i+1
        distance_info = (i, j, dist)
        results.append(distance_info)
    if save_to_file:
        file_result = np.zeros(shape=(vec_count, vec_count))
        for row, col, distance in results:
            file_result[row][col] = distance
            file_result[col][row] = distance
        np.savetxt(save_to_file, file_result, delimiter=',', fmt=['%s']*file_result.shape[1], newline='\n')
    if save_to_figure:
        # 三维直方图
        pass
    return results


def mean_distances(n_vec, algorithm_type, save_to_file='', save_to_figure=''):
    """

    :param n_vec:
    :param algorithm_type:
    :param save_to_file:
    :param save_to_figure:
    :return:
    """
    mean = np.mean(n_vec, axis=0)
    results = []
    for vec in n_vec:
        distance = ALGORITHM_FUNC[algorithm_type](vec, mean)
        results.append(distance)
    if save_to_file:
        np.savetxt(save_to_file, results, delimiter=',', fmt='%.15f', newline='\n')
    if save_to_figure:
        plt.hist(results, bins=20)
        plt.xlabel('number')
        plt.ylabel('euclidean distance')
        plt.savefig(save_to_figure)
        plt.show()
    return results


if __name__ == '__main__':
    import data_model
    data = data_model.get_frequency_data()
    two_by_two_distances(data, algorithm_type='euclidean', save_to_file='../results/euclidean_distances_mean.csv',
                         save_to_figure='euclidean_distances_mean.png')
    # mean_distances(data[:100], algorithm_type='euclidean', save_to_file='euclidean_distances_combine.csv',
    #                          save_to_figure='euclidean_distances_combine.png')
