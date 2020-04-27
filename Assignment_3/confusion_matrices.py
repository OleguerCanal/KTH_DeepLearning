
import numpy as np
import sys, pathlib
from helper import read_names, read_names_countries
import cv2
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.metrics import Accuracy
from mlp.models import Sequential
from mlp.losses import CrossEntropy
from mlp.layers import Conv2D, Dense, Softmax, Relu, Flatten, Dropout, MaxPool2D
from mlp.callbacks import MetricTracker, BestModelSaver, LearningRateScheduler
from mlp.utils import plot_confusion_matrix

np.random.seed(1)

if __name__ == "__main__":
    # Load data
    x_train, y_train, x_val, y_val, _, _ = read_names(n_train=-1)
    print(x_train.shape)
    classes = read_names_countries()

    # Load model model
    model = Sequential(loss=CrossEntropy())
    # model.load("models/names_test")
    model.load("models/names_no_compensation")

    y_pred_train = model.predict_classes(x_train)
    y_pred_val = model.predict_classes(x_val)
    
    plot_confusion_matrix(y_pred_train, y_train, classes, "figures/conf_no_compensation_train")
    plot_confusion_matrix(y_pred_val, y_val, classes, "figures/conf_no_compensation_val")
