import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("l2reg_optimization/evaluations.csv")

lr = df[["l2_reg"]].to_numpy()
values = df[["value"]].to_numpy()

plt.scatter(lr, values)
plt.xlabel("l2 regularization")
plt.ylabel("Top Validation Accuracy")
plt.title("Gaussian Process Regression Optimization Evaluations")
plt.show()