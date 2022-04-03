import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

ORI_DATA_PATH = '../cnnhealth.txt'
PROCESSED_DATA_PATH = '../results/word_frequency.txt'


def read_data_from_file():
    if not os.path.exists(ORI_DATA_PATH):
        return

    msg_data = []
    f = open(ORI_DATA_PATH, 'r', encoding='utf8')

    for index, line in enumerate(f.readlines()):
        line = line.strip()
        info_list = line.split('|')
        if len(info_list) >= 3:
            msg_info = info_list[2]
            url_index = msg_info.find('http')
            if url_index != -1:
                msg_info = msg_info[:url_index]
            msg_data.append(msg_info)
    f.close()
    return msg_data


def get_frequency_data():
    msg_data = read_data_from_file()
    vector = CountVectorizer()
    word_frequency_info = vector.fit_transform(msg_data)
    f_data = word_frequency_info.toarray()
    return f_data


def save_frequency_data(f_data):
    np.savetxt(PROCESSED_DATA_PATH, f_data, fmt='%.0f', newline='\n')


if __name__ == '__main__':
    data = get_frequency_data()
    save_frequency_data(data)
