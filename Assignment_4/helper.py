import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.utils import one_hotify


def load_data(path = "data/goblet_book.txt"):
    with open(path, 'r') as file:
        data = list(file.read())
    ind_to_char = list(set(data))  # == book_chars
    char_to_ind = {}
    for ind, elem in enumerate(ind_to_char):
        char_to_ind[elem] = ind
    # for key, val in char_to_ind.items():
    #     assert(ind_to_char[val] == key)
    encoded_data = [char_to_ind[elem] for elem in data]
    encoded_data = one_hotify(np.array(encoded_data), num_classes = len(ind_to_char))
    return encoded_data, ind_to_char, char_to_ind

if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data()
    print(encoded_data)