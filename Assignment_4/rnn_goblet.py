import numpy as np
import sys, pathlib
from helper import load_data
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import VanillaRNN
from mlp.batchers import RnnBatcher
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import one_hotify

np.random.seed(0)

if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/goblet_book.txt")

    # Define callbacks
    # mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    # callbacks = [mt]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 20  # n

    batcher = RnnBatcher(seq_length)

    # # Define model
    v_rnn = VanillaRNN(state_size=100, input_size=K, output_size=K)
    model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    model.add(v_rnn)

    # x = (char_to_ind['.'], )
    # x = one_hotify(x, num_classes = K)
    # print(x.shape)
    # for i in range(10):
    #     probs = v_rnn(x)
    #     print(probs.shape)
    #     probs = probs.flatten()
    #     # next_elem = np.random.choice(list(range(len(probs))), probs)
    #     next_elem = np.argmax(probs)
    #     print(probs)
    #     print(ind_to_char[next_elem])
    #     x = one_hotify(next_elem, K)

    # # Fit model
    model.fit(X=encoded_data, epochs=2, lr = 1e-2, momentum=0.95,
              batcher=batcher, callbacks=[])
    # model.save("models/names_best")

    # mt.plot_training_progress(save=True, name="figures/names_best")


    x = np.array((char_to_ind['i'], char_to_ind['s'], char_to_ind[' '], ))
    x = one_hotify(x, num_classes = K)
    pred = [ind_to_char[elem] for elem in np.argmax(x, axis=0)]
    print(pred)
    print(x)
    print(x.shape)
    for i in range(10):
        probs = v_rnn(x)
        next_elem = np.argmax(probs, axis=0)
        pred = [ind_to_char[elem] for elem in next_elem]
        print(pred)
        x = one_hotify(next_elem, K)