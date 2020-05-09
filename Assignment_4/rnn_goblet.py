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
from mlp.utils import one_hotify, generate_sequence

np.random.seed(0)

if __name__ == "__main__":
    # encoded_data, ind_to_char, char_to_ind = load_data(path="data/test.txt")
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/goblet_book.txt")

    # Define callbacks
    mt = MetricTracker(frequency = 100)  # Stores training evolution info (losses and metrics)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 20  # n

    # Define model
    v_rnn = VanillaRNN(state_size=100, input_size=K, output_size=K)
    model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    model.add(v_rnn)

    # Fit model
    batcher = RnnBatcher(seq_length)
    model.fit(X=encoded_data, epochs=3, lr = 1e-1, momentum=0.95,
              batcher=batcher, callbacks=callbacks)
    model.save("models/rnn_goblet")

    # Analyse results
    mt.plot_training_progress(save=True, name="figures/goblet")
    v_rnn.reset_state()
    generate_sequence(v_rnn, 'h', ind_to_char, char_to_ind, length=20)