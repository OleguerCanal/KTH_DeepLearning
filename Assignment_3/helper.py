import numpy as np

def load_idxfile(filename):

    """
    Load idx file format. For more information : http://yann.lecun.com/exdb/mnist/ 
    """
    import struct
    
    filename = "data/" + filename
    with open(filename,'rb') as _file:
        if ord(_file.read(1)) != 0 or ord(_file.read(1)) != 0 :
           raise Exception('Invalid idx file: unexpected magic number!')
        dtype,ndim = ord(_file.read(1)),ord(_file.read(1))
        shape = [struct.unpack(">I", _file.read(4))[0] for _ in range(ndim)]
        data = np.fromfile(_file, dtype=np.dtype(np.uint8).newbyteorder('>')).reshape(shape)
    return data
    
def read_mnist(dim=[28,28],n_train=50000, n_val=10000,n_test=1000):

    """
    Read mnist train and test data. Images are normalized to be in range [0,1]. Labels are one-hot coded.
    """    
    import scipy.misc

    train_imgs = load_idxfile("train-images-idx3-ubyte")
    train_imgs = train_imgs / 255.
    train_imgs = train_imgs.reshape(-1,dim[0]*dim[1])

    train_lbls = load_idxfile("train-labels-idx1-ubyte")
    train_lbls_1hot = np.zeros((len(train_lbls),10),dtype=np.float32)
    train_lbls_1hot[range(len(train_lbls)),train_lbls] = 1.

    test_imgs = load_idxfile("t10k-images-idx3-ubyte")
    test_imgs = test_imgs / 255.
    test_imgs = test_imgs.reshape(-1,dim[0]*dim[1])

    test_lbls = load_idxfile("t10k-labels-idx1-ubyte")
    test_lbls_1hot = np.zeros((len(test_lbls),10),dtype=np.float32)
    test_lbls_1hot[range(len(test_lbls)),test_lbls] = 1.

    def rs(imgs):
        imgs = (imgs.T).reshape((dim[0], dim[1], imgs.shape[0]), order='C')
        return np.expand_dims(imgs, axis=2).astype(float)  # h, w, c, n
    return rs(train_imgs[:n_train]),train_lbls_1hot[:n_train].T.astype(float),\
           rs(train_imgs[n_train:n_train+n_val]),train_lbls_1hot[n_train:n_train+n_val].T.astype(float),\
           rs(test_imgs[:n_test]),test_lbls_1hot[:n_test].T.astype(float)

def read_cifar_10(n_train=None, n_val=None, n_test=None):
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")
    from mlp.utils import LoadXY

    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    for i in [2, 3, 4, 5]:
        x, y = LoadXY("data_batch_" + str(i))
        x_train = np.concatenate((x_train, x), axis=1)
        y_train = np.concatenate((y_train, y), axis=1)
    x_val = x_train[:, -1000:]
    y_val = y_train[:, -1000:]
    x_train = x_train[:, :-1000]
    y_train = y_train[:, :-1000]
    x_test, y_test = LoadXY("test_batch")

    if n_train is not None:
        x_train = x_train[..., 14000:14000+n_train]
        y_train = y_train[..., 14000:14000+n_train]
    if n_val is not None:
        x_val = x_val[..., 0:n_val]
        y_val = y_val[..., 0:n_val]
    if n_test is not None:
        x_test = x_test[..., 0:n_test]
        y_test = y_test[..., 0:n_test]

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # reshaped_train = np.zeros((32, 32, 3, x_train.shape[-1]))
    # for i in range(x_train.shape[-1]):
    #     flatted_image = np.array(x_train[..., i])
    #     image = np.reshape(flatted_image,  (32, 32, 3), order='F')
    #     cv2.imshow("image", image)
    #     cv2.waitKey()

    x_train = np.reshape(np.array(x_train), (32, 32, 3, x_train.shape[-1]), order='F')
    x_val = np.reshape(np.array(x_val), (32, 32, 3, x_val.shape[-1]), order='F')
    x_test = np.reshape(np.array(x_test), (32, 32, 3, x_test.shape[-1]), order='F')

    return x_train.astype(float), y_train.astype(float), x_val.astype(float), y_val.astype(float), x_test.astype(float), y_test.astype(float)

def read_names(n_train=-1):
    def read_file(filepath="data/names/ascii_names.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = -1
        for name in names:
            n_len = max(n_len, len(name))
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)

    val_indxs = []
    with open("data/names/Validation_Inds.txt") as fp:
        val_indxs = [int(val) for val in fp.readline().split(" ")]

    indx = list(range(len(names)))
    np.random.shuffle(indx)
    for val in val_indxs:
        indx.remove(val)

    return x[..., indx[:n_train]], y[..., indx[:n_train]],\
           x[..., val_indxs], y[..., val_indxs],\
           None, None


def read_names_test(n_train=-1):
    def read_file(filepath="data/names/test.txt"):
        names = []
        labels = []
        with open(filepath) as fp:
            line = fp.readline()
            while line:
                line = line.replace("  ", " ")
                # print(line)
                names.append(line.split(" ")[0].lower())
                labels.append(int(line.split(" ")[-1]))
                line = fp.readline()
        return names, labels

    def encode_names(names):
        n_len = 19
        x = np.zeros((ord('z')-ord('a')+1, n_len, len(names)))
        for n in range(len(names)):
            for i, char in enumerate(names[n]):
                if ord(char) > ord('z') or ord(char) < ord('a'):
                    continue
                x[ord(char)-ord('a')][i][n] = 1
        return np.expand_dims(x, axis=2).astype(float)        

    def get_one_hot_labels(labels):
        labels = np.array(labels)
        one_hot_labels = np.zeros((labels.size, np.max(labels)))
        one_hot_labels[np.arange(labels.size), labels-1] = 1
        return one_hot_labels.T
    
    names, labels = read_file()
    x = encode_names(names)
    y = get_one_hot_labels(labels)
    return x, y, names

def read_names_countries():
    return ["Arabic", "Chinese", "Czech", "Dutch", "English", "French", "German",\
            "Greek", "Irish", "Italian", "Japanese", "Korean", "Polish", "Portuguese",\
            "Russian", "Scottish", "Spanish", "Vietnamese"]