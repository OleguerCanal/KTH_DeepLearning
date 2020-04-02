# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import matplotlib.pyplot as plt

from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CrossEntropy
from mlp.models import Sequential
from mlp.metrics import accuracy
from mlp.utils import LoadXY, prob_to_class
from mpo.metaparamoptimizer import MetaParamOptimizer
from util.misc import dict_to_string

np.random.seed(0)

def montage(W, title, path=None):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2,5)
    plt.suptitle(title)
    for i in range(2):
        for j in range(5):
            im  = W[5*i+j,:].reshape(32,32,3, order='F')
            sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
            sim = sim.transpose(1,0,2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y="+str(5*i+j))
            ax[i][j].axis('off')
    if path is not None:
        directory = "/".join(path.split("/")[:-1])
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + ".png")
    else:
        plt.show()
    

# Define evaluator (function to run in MetaParamOptimizer)
def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_path="", **kwargs):
    # Define model
    model = Sequential(loss=CrossEntropy())
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))
    model.add(Softmax())

    # Fit model
    model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val, **kwargs)
    test_acc = model.get_classification_metrics(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) + ", Test Acc: " + str(test_acc)
    model.plot_training_progress(show=False,
                                save=True,
                                name="figures/param_testing/" + dict_to_string(kwargs),
                                subtitle=subtitle)
    model.save(experiment_path + "/" + dict_to_string(kwargs))
    montage(W=np.array(model.layers[0].weights[:, :-1]),
            title=subtitle,
            path="figures/param_testing/weights/" + dict_to_string(kwargs) + "_weights")

    # Minimizing value: validation accuracy
    val_acc = model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": model}  # Save score and model
    return result

if __name__ == "__main__":
    # Download & Extract CIFAR-10 Python (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
    # Put it in a Data folder

    # Load data
    x_train, y_train = LoadXY("data_batch_1")
    x_val, y_val = LoadXY("data_batch_2")
    x_test, y_test = LoadXY("test_batch")

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    # Define list of parameters to try
    dicts_list = [
        { "l2_reg": 0.0, "lr": 0.1 },
        { "l2_reg": 0.0, "lr": 0.001 },
        { "l2_reg": 0.1, "lr": 0.001 },
        { "l2_reg": 1.0, "lr": 0.001 },
    ]
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_path" : "models/param_testing/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "x_test" : x_test,
        "y_test" : y_test,
        "batch_size": 100,
        "epochs" : 40,
        "momentum" : 0.,
        "shuffle_minibatch" : False,
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path=fixed_args["experiment_path"])
    best_model = mpo.list_search(evaluator=evaluator,
                                dicts_list=dicts_list,
                                fixed_args=fixed_args)
    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)
