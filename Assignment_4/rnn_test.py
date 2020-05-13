import numpy as np
import sys, pathlib
from helper import load_data
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import VanillaRNN, Softmax
from mlp.batchers import RnnBatcher
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import one_hotify, generate_sequence

np.random.seed(0)

if __name__ == "__main__":
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/test_2.txt")
    # encoded_data, ind_to_char, char_to_ind = load_data(path="data/goblet_book.txt")

    # Define callbacks
    mt = MetricTracker(frequency = 100)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    # callbacks = [mt, lrs]
    callbacks = [mt]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 10  # n

    # Define model
    v_rnn = VanillaRNN(state_size=10, input_size=K, output_size=K)
    model = Sequential(loss=CrossEntropy(class_count=None), metric=Accuracy())
    model.add(v_rnn)
    model.add(Softmax())

    # Fit model
    batcher = RnnBatcher(seq_length, ind_to_char)
    model.fit(X=encoded_data, epochs=100, lr = 0.1,
              batcher=batcher, callbacks=callbacks)
    model.save("models/rnn_test")

    mt.plot_training_progress(show=False, save=True, name="figures/rnn_test")
    
    v_rnn.reset_state()
    seq = generate_sequence(v_rnn, 'I', ind_to_char, char_to_ind, length=100)
    print(seq)