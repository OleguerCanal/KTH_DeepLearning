import numpy as np
import sys, pathlib
import cv2

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

from mlp.callbacks import MetricTracker

mt = MetricTracker()
mt.metric_name = "Accuracy"
mt.load("models/tracker")
mt.plot_training_progress()
