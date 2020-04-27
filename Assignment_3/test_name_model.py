
import numpy as np
import sys, pathlib
from helper import read_names, read_names_countries, read_names_test
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
    x_test, y_test, names = read_names_test()
    classes = read_names_countries()

    print(x_test.shape)

    # Load model model
    model = Sequential(loss=CrossEntropy())
    model.load("models/names_test")
    # model.load("models/names_no_compensation")

    y_pred_prob_test = model.predict(x_test)
    y_pred_test = model.predict_classes(x_test)
    print(y_pred_prob_test)
    print(y_test)
    
    plot_confusion_matrix(y_pred_test, y_test, classes, "figures/conf_test")

    import matplotlib.pyplot as plt
    plt.title("Prediction Vectors")      
    pos = plt.imshow(y_pred_prob_test.T)
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.yticks(range(len(names)), names)
    # plt.xticks(rotation=45, ha='right')
    plt.colorbar(pos)
    plt.savefig("figures/prob_vector_test")
    plt.show()