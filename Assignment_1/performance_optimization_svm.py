# Add path to Toy-DeepLearning-Framework
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]) + "/Toy-DeepLearning-Framework/")

import numpy as np
import matplotlib.pyplot as plt

from mlp.layers import Dense, Softmax, Relu
from mlp.losses import CategoricalHinge
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
def evaluator(x_train, y_train, x_val, y_val, x_test, y_test, experiment_name="", init="fixed", **kwargs):
    # Define model
    model = Sequential(loss=CategoricalHinge())
    model.add(Dense(nodes=10, input_dim=x_train.shape[0]))

    # Fit model
    model_save_path = "models/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init
    best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
                           save_path=model_save_path, **kwargs)
    
    # Plot results
    test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) +\
                ", weight_init:" + init + ", Test Acc: " + str(test_acc)
    best_model.plot_training_progress(show=False,
                                    save=True,
                                    name="figures/" + experiment_name + "/" + dict_to_string(kwargs) + "_" + init,
                                    subtitle=subtitle)
    montage(W=np.array(best_model.layers[0].weights[:, :-1]),
            title=subtitle,
            path="figures/" + experiment_name + "/weights/" + dict_to_string(kwargs) + "_" + init)

    # Minimizing value: validation accuracy
    val_acc = best_model.get_classification_metrics(x_val, y_val)[0] # Get accuracy
    result = {"value": val_acc, "model": best_model}  # Save score and model
    return result

if __name__ == "__main__":
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

    # Preprocessing
    mean_x = np.mean(x_train)
    std_x = np.std(x_train)
    x_train = (x_train - mean_x)/std_x
    x_val = (x_val - mean_x)/std_x
    x_test = (x_test - mean_x)/std_x

    search_space = {  # Optimization will be performed on all combinations of these
        "batch_size": [20, 100, 400],     # Batch sizes
        "lr": [0.0001, 0.001, 0.01],      # Learning rates
        "l2_reg": [0.01, 0.05, 0.1],      # L2 Regularization terms
        "momentum" : [0.5, 0.7],
    }
    # Define fixed params (constant through optimization)
    fixed_args = {
        "experiment_name" : "performance_optimization_svm/",
        "x_train" : x_train,
        "y_train" : y_train,
        "x_val" : x_val,
        "y_val" : y_val,
        "x_test" : x_test,
        "y_test" : y_test,
        "epochs" : 120,
        "init": "normal"
    }
    # NOTE: The union of both dictionaries should contain all evaluator parameters

    # Perform optimization
    mpo = MetaParamOptimizer(save_path="models/" + fixed_args["experiment_name"])
    best_model = mpo.grid_search(evaluator=evaluator,
                                search_space=search_space,
                                fixed_args=fixed_args)

    # TESTING
    # model = Sequential(loss="cross_entropy")
    # model.add(
    #     Dense(nodes=10, input_dim=x_train.shape[0], weight_initialization="fixed"))
    # model.add(Activation("softmax"))
    # best_model = model.fit(X=x_train, Y=y_train, X_val=x_val, Y_val=y_val,
    #                     batch_size=50, epochs=10, lr=0.01, # 0 lr will not change weights
    #                     momentum=0.5, l2_reg=0.01, save_path="models/performance_optimization/test1")
    # test_acc = best_model.get_classification_metrics(x_test, y_test)[0]
    # # subtitle = "l2_reg: " + str(kwargs["l2_reg"]) + ", lr: " + str(kwargs["lr"]) + ", Test Acc: " + str(test_acc)
    # best_model.plot_training_progress(show=False,
    #                             save=True,
    #                             name="figures/performance_optimization/test",
    #                             subtitle="subtitle")
    # val_acc = best_model.get_classification_metrics(x_val, y_val)[0]
    # print("test_acc:", test_acc)
    # print("val_acc:", val_acc)
    
    # Test model
    test_acc, test_loss = best_model["model"].get_classification_metrics(x_test, y_test)
    print("Test accuracy:", test_acc)
