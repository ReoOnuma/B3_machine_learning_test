import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# load datasets
iris = datasets.load_iris()
# load as pandas.DataFrame
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# add target
df_iris['target'] = iris.target
# split the dataset
data_train, data_test, target_train, target_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# define neural network model
clf = MLPClassifier(hidden_layer_sizes=10, activation='relu', solver='adam', max_iter=1000)

