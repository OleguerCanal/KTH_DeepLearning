import numpy as np
import sys, pathlib
from helper import load_data
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import VanillaRNN
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import one_hotify

np.random.seed(0)

if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data()

    # Define callbacks
    # mt = MetricTracker()  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    # callbacks = [mt]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 25  # n

    v_rnn = VanillaRNN(state_size=100, input_size=K, output_size=K, seq_length=1)


    x = (char_to_ind['.'], )
    x = one_hotify(x, num_classes = K)
    print(x.shape)
    for i in range(10):
        probs = v_rnn(x)
        probs = probs.flatten()
        # next_elem = np.random.choice(list(range(len(probs))), probs)
        next_elem = np.argmax(probs)
        print(probs)
        print(ind_to_char[next_elem])
        x = one_hotify(next_elem, K)

    # # Define model
    # model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    # model.add(VanillaRNN(state_size=100, output_size=K, seq_length=seq_length))

    # # Fit model
    # model.fit(X=encoded_data,
    #           batch_size=100, epochs=500, lr = 1e-3, momentum=0.8, l2_reg=0.001,
    #           compensate=True, callbacks=callbacks)
    # model.save("models/names_best")

    # mt.plot_training_progress(save=True, name="figures/names_best")