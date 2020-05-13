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
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler, TextSynthesiser
from mlp.utils import one_hotify, generate_sequence

np.random.seed(0)

experiment_name = "rnn_goblet_17"


if __name__ == "__main__":
    # encoded_data, ind_to_char, char_to_ind = load_data(path="data/test.txt")
    encoded_data, ind_to_char, char_to_ind = load_data(path="data/goblet_book.txt")

    # Define callbacks
    mt = MetricTracker(file_name="models/tracker_" + experiment_name, frequency = 1000)  # Stores training evolution info (losses and metrics)
    ts = TextSynthesiser(ind_to_char, char_to_ind, file_name="synt_text/" + experiment_name)
    # lrs = LearningRateScheduler(evolution="constant", lr_min=1e-3, lr_max=9e-1)
    callbacks = [mt, ts]

    # Define hyperparams
    K = len(ind_to_char)
    seq_length = 25  # n

    # Define model
    model = Sequential(loss=CrossEntropy(average=False), metric=Accuracy())
    model.add(VanillaRNN(state_size=200, input_size=K, output_size=K))
    model.add(Softmax())
    # model.load("models/rnn_goblet_14")

    # Fit model
    batcher = RnnBatcher(seq_length)
    model.fit(X=encoded_data, epochs=50, lr = 1e-1,
              batcher=batcher, callbacks=callbacks)
    model.save("models/" + experiment_name)

    # Analyse results
    mt.plot_training_progress(show=False, save=True, name="figures/" + experiment_name)
    string = generate_sequence(model.layers[0], '.', ind_to_char, char_to_ind, length=1000)
    with open("synt_text/" + experiment_name + "_seq_final", "w") as text_file:
        text_file.write(string)