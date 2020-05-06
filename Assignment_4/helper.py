import numpy as np

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
    return encoded_data, ind_to_char, char_to_ind

if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data()
    print(encoded_data)