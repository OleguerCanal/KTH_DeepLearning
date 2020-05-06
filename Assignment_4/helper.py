import numpy as np

def encode_text(path = "data/goblet_book.txt"):
    with open(path, 'r') as file:
        data = list(file.read())
    ind_to_char = list(set(data))
    char_to_ind = {}
    for ind, elem in enumerate(ind_to_char):
        char_to_ind[elem] = ind
    # for key, val in char_to_ind.items():
    #     assert(ind_to_char[val] == key)
    return ind_to_char, char_to_ind

if __name__ == "__main__":
    ind_to_char, char_to_ind = encode_text()